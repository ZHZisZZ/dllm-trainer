"""
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=12 --time=03:00:000 \
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/roberta/pt.py \
        --model_name_or_path "FacebookAI/roberta-large" \
        --dataset_args "Trelis/tiny-shakespeare" \
        --text_field "Text" \
        --max_length 256 \
        --output_dir "models/roberta-large/tiny-shakespeare"

srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:8 --cpus-per-task=12 --time=03:00:000 \
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/roberta/pt.py \
        --model_name_or_path "microsoft/deberta-v2-xxlarge" \
        --dataset_args "wikitext[name:wikitext-103-v1]" \
        --text_field "text" \
        --max_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --output_dir "models/deberta-v2-xxlarge/wikitext-103-v1"


# wikitext-103-v1
        
sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "ddp" \
    --script_path "examples/roberta/pt.py" \
    --model_name_or_path "FacebookAI/roberta-large" \
    --dataset_args "wikitext[name:wikitext-103-v1]" \
    --text_field "text" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/roberta-large/wikitext-103-v1"

sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "ddp" \
    --script_path "examples/roberta/pt.py" \
    --model_name_or_path "FacebookAI/roberta-base" \
    --dataset_args "wikitext[name:wikitext-103-v1]" \
    --text_field "text" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/roberta-base/wikitext-103-v1"


# openwebtext

sbatch --nodes=8 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "ddp" \
    --script_path "examples/roberta/pt.py" \
    --model_name_or_path "FacebookAI/roberta-large" \
    --dataset_args "dylanebert/openwebtext" \
    --text_field "text" \
    --streaming True \
    --insert_eos True \
    --max_steps 60000 \
    --max_length 512 \
    --per_device_train_batch_size 64 \
    --eval_strategy "no" \
    --eval_on_start False \
    --save_steps 0.05 \
    --output_dir "models/roberta-large/openwebtext"

sbatch --nodes=16 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "ddp" \
    --script_path "examples/roberta/pt.py" \
    --model_name_or_path "FacebookAI/roberta-large" \
    --dataset_args "dylanebert/openwebtext" \
    --text_field "text" \
    --streaming True \
    --insert_eos True \
    --max_steps 50000 \
    --max_length 512 \
    --per_device_train_batch_size 64 \
    --eval_strategy "no" \
    --eval_on_start False \
    --save_steps 0.05 \
    --output_dir "models/roberta-large/openwebtext"
"""

import os
import functools
from dataclasses import dataclass, field

import torch
import transformers
import accelerate

import dllm


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    # Uses only the configuration from model_name_or_path to initialize the model from scratch
    model_name_or_path: str = (
        "FacebookAI/roberta-large"  # "FacebookAI/roberta-base"
    )

@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "wikitext[name:wikitext-103-v1]"
    text_field: str = "text"
    max_length: int = 256
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(
        default=False,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    random_length_ratio: float = field(
        default=0.0,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#pre-training for reference."
            )
        },
    )

@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/roberta-large/wikitext-103-v1"
    learning_rate: float = 1e-4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    eval_steps: float = 0.1
    save_steps: float = 0.1


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
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args, streaming=data_args.streaming)
        dataset = dataset.map(
            functools.partial(
                dllm.utils.tokenize_and_group, 
                tokenizer=tokenizer, 
                text_field=data_args.text_field, 
                seq_length=data_args.max_length, 
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    # ----- Training --------------------------------------------------------------
    @dataclass
    class Collator(transformers.DataCollatorForSeq2Seq):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#pre-training
        # By default, 1% of the pre-training data are truncated to a random length
        random_length_ratio: float = 0.01

        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors)
            if torch.rand(1) < self.random_length_ratio:
                random_length = torch.randint(
                    1, outputs["input_ids"].shape[1] + 1, (1,)
                )
                for key in ["input_ids", "labels", "attention_mask"]:
                    if key in outputs: outputs[key] = outputs[key][:, :random_length]
            # Check if attention_mask is all ones and set it to None
            if torch.all(outputs["attention_mask"] == 1):
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
        data_collator=Collator(
            tokenizer,
            return_tensors="pt",
            padding=True,
            random_length_ratio=data_args.random_length_ratio,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
