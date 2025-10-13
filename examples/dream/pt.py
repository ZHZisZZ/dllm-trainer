"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/dream/pt.py

- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/dream/pt.py

Slurm users
------------
# Note: run `mkdir logs` before running sbatch; adjust partition & quotatype as needed.
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/dream/pt.py"

- 16 GPUs (2 Nodes, ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/dream/pt.py"
"""

import os
from dataclasses import dataclass, field
import torch
import transformers
import accelerate
import datasets

import dllm
from dllm.pipelines import dream


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/Dream-7B-PT"
    learning_rate: float = 3e-4
    max_steps: int = 2_000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_steps: float = 0.05
    save_steps: float = 0.05
    # Dream PT specific args
    # Note: Since Dream’s pretraining recipe is not public,
    # this is only a reference implementation following LLaDA’s data processing approach.
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md."
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
    # ----- Parse & setup --------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.accelerator_config.dispatch_batches = False  # for streaming dataset
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Model ---------------------------------------------------------------
    # Initialize from config (Dream pretraining = from scratch)
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(config, dtype=torch.bfloat16)

    # ----- Tokenizer -----------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA -------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset -------------------------------------------------------------
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

    # ----- Data Collator -------------------------------------------------------
    @dataclass
    class DreamPTCollator(transformers.DataCollatorForSeq2Seq):
        random_length_ratio: float = 0.01
        label_pad_token_id: int = -100

        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors=return_tensors)
            input_ids, labels = outputs["input_ids"], outputs["labels"]
            bsz, seq_len = input_ids.shape

            # --- Random truncation for robustness ---
            if torch.rand(1).item() < self.random_length_ratio:
                random_len = torch.randint(1, seq_len + 1, (1,)).item()
                input_ids = input_ids[:, :random_len]
                labels = labels[:, :random_len]

            # --- Add BOS token to the beginning of input_ids ---
            bos = torch.full(
                (bsz, 1),
                self.tokenizer.bos_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([bos, input_ids], dim=1)

            # --- Prepend zeros to labels instead of BOS ---
            ignore_labels = self.label_pad_token_id * torch.ones(
                (bsz, 1), dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([ignore_labels, labels], dim=1)

            # --- Update and return ---
            outputs["input_ids"] = input_ids
            outputs["labels"] = labels
            return outputs

    # ----- Trainer -------------------------------------------------------------
    trainer = dream.DreamTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        loss_weight_type=training_args.loss_weight_type,
        data_collator=DreamPTCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
            random_length_ratio=training_args.random_length_ratio,
        ),
    )

    # ----- Training Loop -------------------------------------------------------
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
