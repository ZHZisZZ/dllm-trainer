"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/llada/sft.py

- 1 GPU (4bit quant, LoRA) & Weight merging:
    # train
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/llada/sft.py \
        --load_in_4bit True --lora True

    # merge lora weights
    python dllm_trainer/tools/merge_peft_adapter.py \
        --adapter_model_name_or_path models/LLaDA-8B-SFT/checkpoint-final \
        --output_model_name_or_path models/LLaDA-8B-SFT/checkpoint-final-merged \
        --dtype bf16
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/llada/sft.py  --output_dir "models-tmp"

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/llada/sft.py"

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/llada/sft.py"
"""
import os
import functools
from dataclasses import dataclass

import transformers
import accelerate

import dllm
from dllm.pipelines import llada

@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base" # "inclusionAI/LLaDA-MoE-7B-A1B-Base"

@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-8B-SFT"
    # others (llada specific training params)
    mask_prompt_loss: bool = True


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        dllm.utils.DataArguments, 
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    transformers.set_seed(training_args.seed)
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup()

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset ----------------------------------------------------------------
    def train_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_prompt_loss: bool = True, 
        label_pad_token_id: int = -100
    ) -> dict:
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], 
            tokenize=True, 
            add_generation_prompt=False
        )
        labels = prompt_response_tokens.copy()
        if mask_prompt_loss: 
            prompt_tokens = tokenizer.apply_chat_template(
                row["messages"][:-1], 
                tokenize=True, 
                add_generation_prompt=True
            )
            labels[:len(prompt_tokens)] = [label_pad_token_id] * len(prompt_tokens)
        return {"input_ids": prompt_response_tokens, "labels": labels}

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(
            functools.partial(
                train_map_fn, 
                tokenizer=tokenizer,
                mask_prompt_loss=training_args.mask_prompt_loss,
            ), 
            num_proc=data_args.num_proc,
        )
        dataset = dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
        )

    # ----- Training --------------------------------------------------------------
    trainer = llada.LLaDATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt", 
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id, # LLaDA is trained on padding <eos_token>
        )
    )
    trainer.train()
    trainer.save_model(os.path.join(
        training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(os.path.join(
        training_args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    train()
