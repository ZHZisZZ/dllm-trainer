"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/dream/sft.py

- 1 GPU (4bit quant, LoRA) & Weight merging:
    # train
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/dream/sft.py \
        --load_in_4bit True --lora True
    
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

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/dream/sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/dream/sft.py"
"""
import os
import functools
from dataclasses import dataclass

import transformers
import accelerate

import dllm
from dllm.pipelines import dream

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"

@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"

@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/Dream-7B-SFT"
    # others (dream specific training params)
    perbatch_cutoff: bool = True
    resp_cutoff_ratio: float = 0.0
    mask_prompt_loss: bool = True


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False # necessary when batch contains customized fields
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset ----------------------------------------------------------------
    def sft_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_prompt_loss: bool = True, 
        label_pad_token_id: int = -100
    ) -> dict:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], 
            tokenize=True, 
            add_generation_prompt=True
        )
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], 
            tokenize=True, 
            add_generation_prompt=False
        )
        labels = prompt_response_tokens.copy()
        if mask_prompt_loss: labels[:len(prompt_tokens)] = [label_pad_token_id] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "attention_mask": [1.0] * len(prompt_response_tokens),
            "prompt_len": len(prompt_tokens)
        }

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(
            functools.partial(
                sft_map_fn, 
                tokenizer=tokenizer,
                mask_prompt_loss=training_args.mask_prompt_loss,
            ), 
            num_proc=data_args.num_proc,
        )
        dataset = dllm.utils.post_process_dataset(dataset, data_args) # truncate / filter long sequences if needed

    # ----- Training --------------------------------------------------------------
    trainer = dream.DreamTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=dream.utils.DreamSFTCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
            perbatch_cutoff=training_args.perbatch_cutoff,
            resp_cutoff_ratio=training_args.resp_cutoff_ratio,
        )
    )
    trainer.train()
    trainer.save_model(os.path.join(
        training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(os.path.join(
        training_args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    train()
