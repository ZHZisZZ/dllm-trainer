"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/llada/sft.py
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/llada/sft.py"
"""

import os
from dataclasses import dataclass, field

import transformers
import accelerate

import dllm
from dllm.pipelines import llada


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = (
        "GSAI-ML/LLaDA-8B-Base"  # "inclusionAI/LLaDA-MoE-7B-A1B-Base"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-8B-SFT/tulu-3-sft-mixture[train:10000,test:1000]"
    # LLaDA SFT specific args
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        labels = prompt_response_tokens.copy()
        if training_args.mask_prompt_loss:
            prompt_tokens = tokenizer.apply_chat_template(
                row["messages"][:-1], tokenize=True, add_generation_prompt=True
            )
            # -100s in labels indicate positions where tokens should not be masked
            # and loss should be ignored; all other positions match `input_ids`
            labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
            # `prompt_len` helps `post_process_dataset` truncate long sequences properly
            return {
                "input_ids": prompt_response_tokens,
                "labels": labels,
                "prompt_len": len(prompt_tokens),
            }
        return {"input_ids": prompt_response_tokens, "labels": labels}

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(sft_map_fn, num_proc=data_args.num_proc)
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    @dataclass
    class LLaDASFTCollator(transformers.DataCollatorForSeq2Seq):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#sft
        #
        # LLaDA is finetuned on all tokens, including padding (<eos_token>).
        # Therefore, the attention_mask — which normally ignores padding tokens — should be disabled.
        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors)
            outputs.pop("attention_mask")
            return outputs

    trainer = llada.LLaDATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=LLaDASFTCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,  # LLaDA should be finetuned on padding <eos_token>
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
