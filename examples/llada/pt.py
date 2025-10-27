"""
Local users
------------
- 1 GPU (4bit quant & LoRA, useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/llada/pt.py \
        --load_in_4bit True --lora True
    
- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/llada/pt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 24 Nodes, 192 GPUs (FSDP):
    sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/llada/pt.py"
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
        "GSAI-ML/LLaDA-8B-Base"  # "inclusionAI/LLaDA-MoE-7B-A1B-Base"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    text_field: str = "text"
    streaming: bool = True
    drop_tail: bool = True
    insert_eos: bool = field(
        default=True,
        metadata={
            "help": "False when adjacent samples from the datasets are semantically coherent."
        },
    )
    random_length_ratio: float = field(
        default=0.01,
        metadata={
            "help": (
                "The probability of randomly cut sequences during training. "
                "See https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md#pre-training for reference."
            )
        },
    )


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


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # necessary for streaming dataset
    if data_args.streaming: training_args.accelerator_config.dispatch_batches = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

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
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    # pack sequences to fixed length (no padding at all); infinite for training
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args, streaming=data_args.streaming)
        def pack(dataset_):
            return dataset_.map(
            functools.partial(
                dllm.utils.tokenize_and_group, 
                tokenizer=tokenizer, 
                text_field=data_args.text_field, 
                seq_length=data_args.max_length, 
                insert_eos=data_args.insert_eos,
                drop_tail=data_args.drop_tail),
            batched=True,
            remove_columns=dataset_["train"].column_names,
        )
        if not data_args.streaming: dataset = pack(dataset) # trainer will shuffle for us
        elif data_args.insert_eos: dataset = pack(dataset.shuffle(seed=training_args.seed))
        else: dataset = pack(dataset).shuffle(seed=training_args.seed)

    # ----- Training --------------------------------------------------------------
    @dataclass
    class LLaDAPTCollator(transformers.DataCollatorForSeq2Seq):
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
        data_collator=LLaDAPTCollator(
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
