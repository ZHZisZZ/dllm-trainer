from dataclasses import dataclass

import torch
import transformers

import dllm
from examples.editflow import sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = "answerdotai/ModernBERT-large"
    lm_head_key: str = "decoder"
    init_editflow_from_src: bool = True


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    max_length: int = 1024


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = (
        "models/EditFlow-ModernBERT-large-Adapt/tulu-3-sft-mixture[train:10000,test:1000]"
    )
    num_train_epochs: float = 10
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 48
    per_device_eval_batch_size: int = 48
    x0_sampler: str = "masks[length:64]"


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    # Create EditFlow model (bf16 init on CUDA)
    ef_cfg = dllm.pipelines.editflow.EditFlowModernBertConfig.from_pretrained(
        model_args.model_name_or_path, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(ef_cfg)
        # Initialize EditFlow model from the src model: copies backbone & clones lm_head
        if model_args.init_editflow_from_src:
            src_model = transformers.AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, dtype=torch.bfloat16
            )
            dllm.pipelines.editflow.utils.init_editflow_from_src(
                model, src_model, lm_head_key=model_args.lm_head_key
            )
            del src_model
    model = dllm.utils.load_peft(model, model_args)

    editflow_sft.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model=model,
    )
