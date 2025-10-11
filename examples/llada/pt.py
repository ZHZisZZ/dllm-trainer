"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/llada/pt.py
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/llada/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config single_gpu" \
        --script_path "examples/llada/pt.py"

- 24 Nodes, 192 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/llada/pt.py"
"""

import os
from dataclasses import dataclass, field

import torch
import transformers
import accelerate
import datasets

import dllm
from dllm.pipelines import llada


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    # Uses only the configuration from model_name_or_path to initialize the model from scratch
    model_name_or_path: str = (
        "GSAI-ML/LLaDA-8B-Base"  # "inclusionAI/LLaDA-MoE-7B-A1B-Base"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = (
        "models/LLaDA-8B-PT/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    )
    learning_rate: float = 3e-4
    max_steps: int = 2_000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_steps: float = 0.05
    save_steps: float = 0.05
    # llada specific
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md."
            )
        },
    )


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.accelerator_config.dispatch_batches = (
        False  # necessary for streaming dataset
    )
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Model ------------------------------------------------------------------
    # initialize model weights from scratch
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(
            config, dtype=torch.bfloat16, init_params=True
        )

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset ----------------------------------------------------------------
    # pack sequences to fixed length (no padding at all); infinite for training
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(data_args.dataset_args)
        dataset = datasets.IterableDatasetDict(
            {
                split: dllm.utils.ConstantLengthDataset(
                    tokenizer=tokenizer,
                    dataset=dataset[split],
                    dataset_text_field="text",
                    seq_length=data_args.max_length,
                    num_of_sequences=4096,
                    infinite=(split == "train"),
                    append_concat_token=True,
                    add_special_tokens=False,
                )
                for split in dataset.keys()
            }
        )

    # ----- Training --------------------------------------------------------------
    @dataclass
    class LLaDAPTCollator(transformers.DataCollatorForSeq2Seq):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        random_length_ratio: float = 0.01

        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors)
            if torch.rand(1) < self.random_length_ratio:
                random_length = torch.randint(
                    1, outputs["input_ids"].shape[1] + 1, (1,)
                )
                outputs["input_ids"] = outputs["input_ids"][:, :random_length]
                outputs["labels"] = outputs["labels"][:, :random_length]
            # LLaDA is trained on padding <eos_token>
            outputs.pop("attention_mask")
            return outputs

    trainer = llada.LLaDATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=LLaDAPTCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,  # LLaDA is trained on padding <eos_token>
            random_length_ratio=training_args.random_length_ratio,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
