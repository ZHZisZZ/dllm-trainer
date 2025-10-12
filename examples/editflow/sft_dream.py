"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/editflow/sft_dream.py
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/editflow/sft_dream.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/editflow/sft_dream.py"

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/sft_dream.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/sft_dream.py"
"""

from dataclasses import dataclass

import transformers

import examples.editflow.sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = (
        "models/EditFlow-Dream-7B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]/checkpoint-final"
    )


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = (
        "models/EditFlow-Dream-7B-Instruct-SFT/tulu-3-sft-mixture[train:10000,test:1000]"
    )


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    editflow_sft.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
