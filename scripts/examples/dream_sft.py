"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        scripts/examples/dream_sft.py

- 1 GPU (4bit quant, LoRA) & Weight merging:
    # train
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        scripts/examples/dream_sft.py \
        --load_in_4bit True --lora True

    # merge lora weights
    python dllm_trainer/tools/merge_peft_adapter.py \
        --adapter_model_name_or_path models/Dream-7B-SFT/checkpoint-final \
        --output_model_name_or_path models/Dream-7B-SFT/checkpoint-final-merged \
        --dtype bf16
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        scripts/examples/dream_sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "scripts/examples/dream_sft.py"

- 1 GPU (4bit quant, LoRA):
    sbatch scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "scripts/examples/dream_sft.py" \
        --script_args "--load_in_4bit True --lora True"

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "scripts/examples/dream_sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "scripts/examples/dream_sft.py"
"""
import os
import functools
from dataclasses import dataclass

import torch
import transformers
import accelerate
import peft

import dllm
from dllm.pipelines import dream

@dataclass
class ModelArguments:
    model_name_or_path:     str = "Dream-org/Dream-v0-Base-7B"
    load_in_4bit:           bool = False

@dataclass
class DataArguments:
    dataset_args: str = "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    num_proc: int = 8
    max_length: int = 512

@dataclass
class PeftArguments:
    lora:           bool  = False
    target_modules: str   = "all-linear"
    r:              int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05
    bias:           str   = "none"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "models/Dream-7B-SFT"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    num_train_epochs: float = 3
    logging_steps: float = 10
    eval_on_start: bool = True
    eval_strategy: str = "steps"
    eval_steps: float = 0.25
    save_steps: float = 0.25
    save_only_model: bool = True
    # others (dream specific training params)
    mask_prompt_loss: bool = True


def train():
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        PeftArguments, 
        DataArguments, 
        TrainingArguments
    ))
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # loading model and tokenizer
    try:
        from transformers import modeling_utils as _mu
        def _noop(*args, **kwargs): 
            return
        _mu.caching_allocator_warmup = _noop
    except Exception:
        pass
    model_name_or_path = dllm.utils.resolve_with_base_env(
        model_args.model_name_or_path, "BASE_MODELS_DIR")
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        **(
            {"device_map": {"": accelerate.PartialState().local_process_index}}
            if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
            else {}
        ),
        quantization_config=(
            transformers.BitsAndBytesConfig(load_in_4bit=True) 
            if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available() 
            else None
        ),
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token

    # peft
    if peft_args.lora:
        peft_config = peft.LoraConfig(
            r=peft_args.r,
            target_modules=peft_args.target_modules,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.bias,
        )
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    ################
    # Dataset
    ################
    def train_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_prompt_loss: bool = False, 
        label_pad_token_id: int = -100
    ) -> dict:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], 
            tokenize=True,
            add_generation_prompt=True)
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False)
        # overwrite "<|im_end|>\n" to "<|im_end|><|endoftext|>"
        prompt_response_tokens[-1] = tokenizer.eos_token_id
        labels = prompt_response_tokens.copy()
        if mask_prompt_loss:
            labels[:len(prompt_tokens)] = [label_pad_token_id] * len(prompt_tokens)
        attention_mask = [1.0] * len(prompt_response_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }

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

    ################
    # Training
    ################
    trainer = dream.DreamTrainer(
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
            label_pad_token_id=-100 # within dream training, the padding tokens are not visible and counted
        )
    )
    trainer.train()
    trainer.model.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    train()
