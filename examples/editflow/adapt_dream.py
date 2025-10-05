"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/editflow/adapt_dream.py
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/editflow/adapt_dream.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/editflow/adapt_dream.py"

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/adapt_dream.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/adapt_dream.py"
"""
from dataclasses import dataclass

import transformers

import dllm
import adapt as editflow_adapt


@dataclass
class ModelArguments(editflow_adapt.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    lm_head_key: str = "lm_head"

@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"

@dataclass
class TrainingArguments(editflow_adapt.TrainingArguments):
    output_dir: str = "models/EditFlow-Dream-Instruct-7B/tulu-3-sft-mixture[train:10000,test:1000]"


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        DataArguments, 
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    editflow_adapt.train(
        model_args=model_args, 
        data_args=data_args, 
        training_args=training_args,
        ef_config_cls=dllm.pipelines.editflow.EditFlowDreamConfig,
        ef_model_cls=dllm.pipelines.editflow.EditFlowDreamModel,
    )
