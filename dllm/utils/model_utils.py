import torch
import accelerate
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.utils import disable_caching_allocator_warmup, print_main, load_peft
from dllm.utils.configs import ModelArguments, TrainingArguments


def get_model(
    model_args,
    config: transformers.PretrainedConfig | None = None,
) -> transformers.PreTrainedModel:
    """
    Load a model with flexible input sources.

    Args:
        model_args: An optional dataclass or namespace containing model parameters.
        model_name_or_path: Optional direct model path or name (overrides model_args.model_name_or_path).
        dtype: Dtype (string or torch.dtype).
        load_in_4bit: Whether to load using 4-bit quantization (can override model_args.load_in_4bit).

    Returns:
        transformers.PreTrainedModel
    """

    # Map string dtype to torch dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp": torch.float32,
        "float": torch.float32,
    }

    model_name_or_path = getattr(model_args, "model_name_or_path")
    dtype = getattr(model_args, "dtype", "bfloat16")
    load_in_4bit = getattr(model_args, "load_in_4bit", False)

    # Prefer argument > model_args
    dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)

    # Device map: skip when ZeRO-3
    device_map = (
        {"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        else None
    )

    quant_config = None
    if load_in_4bit and transformers.utils.is_bitsandbytes_available():
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    params = {
        'dtype': dtype,
        'device_map': device_map,
        'quantization_config': quant_config,
        'config': config
    }

    try:
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_name_or_path, **params)
    except:
        model = transformers.AutoModel.from_pretrained(model_name_or_path, **params)

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # Optionally train with lora
    model = load_peft(model, model_args)

    return model


def get_tokenizer(model_args) -> transformers.PreTrainedTokenizer:
    """
    Load a tokenizer with flexible input sources.

    Args:
        model_args: Optional dataclass or namespace containing model parameters.
        model: Optional model instance to configure tokenizer behavior.
        model_name_or_path: Optional direct model name or path (overrides model_args.model_name_or_path).

    Returns:
        transformers.PreTrainedTokenizer
    """
    # Lazy imports to avoid circular dependencies
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.rnd.models.modeling_rnd import RND1LM
    from transformers import BertPreTrainedModel, RobertaPreTrainedModel, ModernBertPreTrainedModel

    model_name_or_path = getattr(model_args, "model_name_or_path")

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )

    assert tokenizer.eos_token != None or tokenizer.pad_token != None

    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token: tokenizer.eos_token = tokenizer.pad_token

    # If model is not provided, return as-is
    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]

    # ---------------- Model-specific customization ----------------
    if issubclass(model_cls, LLaDAModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mdm_mask|>"})
        tokenizer.eot_token = "<|eot_id|>"
        # tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token) # can not do this for llada base directly
        # TODO: for llada base, add special_tokens = {"<|start_header_id|>": 126346, "<|end_header_id|>": 126347, "<|eot_id|>": 126348} 
        # fix bugs in chat template
        tokenizer.chat_template = """\
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor %}
{% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
"""
    elif issubclass(model_cls, LLaDAMoEModelLM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
        tokenizer.eot_token = "<|role_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, DreamModel):
        tokenizer.eot_token = "<|im_end|>"
        tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    elif issubclass(model_cls, RND1LM):
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    elif issubclass(model_cls, (BertPreTrainedModel, RobertaPreTrainedModel, ModernBertPreTrainedModel)):
        tokenizer.eot_token = "[/Answer]"
        tokenizer.chat_template = """\
{% if messages[0]['role'] == 'system' %}
[SYS]
{{ messages[0]['content'] | trim }}
[/SYS]

{% set loop_messages = messages[1:] %}
{% else %}
{% set loop_messages = messages %}
{% endif -%}
{%- for message in loop_messages %}
{% if message['role'] == 'user' %}
[Question]
{{ message['content'] | trim }}
[/Question]

{% elif message['role'] == 'assistant' %}
[Answer]
{{ message['content'] | trim }}
[/Answer]

{% endif %}
{% endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
[Answer]
{% endif %}
"""
    else:
        print_main("no tokenizer customization for model class:", model_cls)
    return tokenizer