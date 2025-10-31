"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/bert/pt.py

- 8 GPUs (DDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml \
        examples/bert/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 8 GPUs (DDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "ddp" \
        --script_path "examples/bert/pt.py"
"""

import os
import functools
from dataclasses import dataclass, field

import transformers
import accelerate

import dllm


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    # Uses only the configuration from model_name_or_path to initialize the model from scratch
    model_name_or_path: str = "FacebookAI/roberta-large"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "wikitext[name:wikitext-2-v1]"
    max_length: int = 512
    text_field: str = "text"
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = field(
        default=False,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    load_preprocessed_data: bool = False

@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/roberta-large/wikitext-2-v1"
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
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args, 
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            dataset = dataset.map(
                functools.partial(
                    dllm.utils.tokenize_and_group, 
                    tokenizer=tokenizer, 
                    text_field=data_args.text_field, 
                    seq_length=data_args.max_length, 
                    insert_eos=data_args.insert_eos,
                    drop_tail=data_args.drop_tail),
                batched=True,
                num_proc=None if data_args.streaming else data_args.num_proc,
                remove_columns=dataset["train"].column_names,
            )
        if data_args.streaming: dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("start training...")
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
