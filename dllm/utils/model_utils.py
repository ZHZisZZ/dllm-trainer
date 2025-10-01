import torch
import accelerate
import transformers

from dllm.utils.configs import ModelArguments, TrainingArguments


def get_model(
    model_args: ModelArguments, 
    training_args: TrainingArguments
) -> transformers.PreTrainedModel:
    # Map string dtype to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    torch_dtype = None
    if getattr(model_args, "torch_dtype", None):
        torch_dtype = dtype_map.get(str(model_args.torch_dtype).lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported torch_dtype: {model_args.torch_dtype}")

    model = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
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
    return model


def get_tokenizer(model_args: ModelArguments) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
