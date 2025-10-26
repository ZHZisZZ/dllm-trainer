"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/sft.py \
        --load_in_4bit True --lora True
    
- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Nodes, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/sft.py"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import transformers
import accelerate

import dllm


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/ModernBERT-large/alpaca"
    group_by_length: bool = True
    learning_rate: float = 1e-4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_steps: float = 0.1
    save_steps: float = 0.1


# -----------------------------
# SFT map function
# -----------------------------
def sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.

    Args:
        row: a dataset row with `messages`
        tokenizer: a HF tokenizer
        mask_prompt_loss: whether to mask prompt tokens (set their labels to -100)

    Returns:
        dict with keys: input_ids, labels, and optionally prompt_len
    """
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)

        # Use functools.partial so this fn can be imported elsewhere without binding tokenizer/flags.
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(map_fn, num_proc=data_args.num_proc)

        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    @dataclass
    class BERTSFTCollator(transformers.DataCollatorForSeq2Seq):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#sft
        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors)
            outputs.pop("attention_mask")
            return outputs

    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=BERTSFTCollator(
            tokenizer,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,  # finetune on padding <eos_token>
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
