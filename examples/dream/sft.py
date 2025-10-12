"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/dream/sft.py
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/dream/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/dream/sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/dream/sft.py"
"""

import os
from dataclasses import dataclass, field

import transformers
import accelerate

import dllm
from dllm.pipelines import dream


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/Dream-7B-SFT"
    # Dream SFT specific args
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    perbatch_cutoff: bool = field(
        default=True,
        metadata={
            "help": (
                "Randomly pick a response length from batch and trim other responses. "
                "See https://github.com/DreamLM/Dream/blob/main/src/trainer/config/sft_trainer.yaml."
            )
        },
    )
    resp_cutoff_ratio: float = field(
        default=0.1,
        metadata={
            "help": (
                "The probability of randomly cutting sequences during training. "
                "See https://github.com/DreamLM/Dream/blob/main/src/trainer/config/sft_trainer.yaml."
            )
        },
    )
    loss_weight_type: str = field(
        default="cart[geo_p:0.3]",
        metadata={
            "help": (
                "The loss weight type. "
                "See https://github.com/DreamLM/Dream/blob/main/src/trainer/config/sft_trainer.yaml."
            )
        },
    )


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset ----------------------------------------------------------------
    def sft_map_fn(row) -> dict:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        labels = prompt_response_tokens.copy()
        if training_args.mask_prompt_loss:
            labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        else:
            # When training on all tokens, prepend a BOS token (if missing)
            # so the model can make predictions for the first mask token.
            if prompt_response_tokens[0] != tokenizer.bos_token_id:
                bos = [tokenizer.bos_token_id]
                prompt_response_tokens = bos + prompt_response_tokens
                prompt_tokens = bos + prompt_tokens
                labels = bos + labels
            labels[0] = -100  # ignore loss on the BOS token
        # `prompt_len` helps `post_process_dataset` truncate long sequences properly
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "attention_mask": [1.0] * len(prompt_response_tokens),
            "prompt_len": len(prompt_tokens),
        }

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(sft_map_fn, num_proc=data_args.num_proc)
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    trainer = dream.DreamTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        loss_weight_type=training_args.loss_weight_type,
        data_collator=dream.utils.DreamSFTCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
            perbatch_cutoff=training_args.perbatch_cutoff,
            resp_cutoff_ratio=training_args.resp_cutoff_ratio,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
