import torch
import accelerate
import transformers
from peft import prepare_model_for_kbit_training

from dllm.utils.utils import disable_caching_allocator_warmup
from dllm.utils.configs import ModelArguments, TrainingArguments


def get_model(
    model_args=None,
    model_name_or_path: str | None = None,
    dtype: str | torch.dtype = "bfloat16",
    load_in_4bit: bool | None = None,
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

    # Prefer argument > model_args
    dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)

    if model_args is not None:
        model_name_or_path = model_name_or_path or getattr(
            model_args, "model_name_or_path", None
        )
        load_in_4bit = load_in_4bit or getattr(model_args, "load_in_4bit", None)

    if not model_name_or_path:
        raise ValueError(
            "`model_name_or_path` must be provided, either directly or via model_args."
        )

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

    model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quant_config,
        config=config,
    )

    # --- if quantized, prepare for LoRA / QLoRA training ---
    if load_in_4bit and quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    return model


def get_tokenizer(
    model_args=None,
    model_name_or_path: str | None = None,
    model: transformers.PreTrainedModel | None = None,
) -> transformers.PreTrainedTokenizer:
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

    # Prefer direct argument > model_args
    if model_args is not None:
        model_name_or_path = model_name_or_path or getattr(
            model_args, "model_name_or_path", None
        )

    if not model_name_or_path:
        raise ValueError(
            "`model_name_or_path` must be provided, either directly or via model_args."
        )

    # ---------------- Tokenizer loading ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # If model is not provided, return as-is
    if not model:
        return tokenizer

    # ---------------- Model-specific customization ----------------

    if isinstance(model, (LLaDAModelLM)):
        tokenizer.mask_token = "<|mdm_mask|>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
        # fix bugs in chat template
        tokenizer.chat_template = """
{% set loop_messages = messages -%}
{%- for message in loop_messages %}
{%- if loop.index0 == 0 -%}{{ bos_token }}{%- endif -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] | trim }}<|eot_id|>
{%- endfor -%}
{%- if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}
""".lstrip()
    elif isinstance(model, (LLaDAMoEModelLM)):
        tokenizer.mask_token = "<|mask|>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    elif isinstance(model, (DreamModel)):
        tokenizer.chat_template = """{%- if tools %}\n {{- '<|im_start|>system\\n' }}\n {%- if messages[0]['role'] == 'system' %}\n {{- messages[0]['content'] }}\n {%- else %}\n {{- 'You are a helpful assistant.' }}\n {%- endif %}\n {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n {%- for tool in tools %}\n {{- \"\\n\" }}\n {{- tool | tojson }}\n {%- endfor %}\n {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' %}\n {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n {%- else %}\n {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n {%- elif message.role == \"assistant\" %}\n {{- '<|im_start|>' + message.role }}\n {%- if message.content %}\n {{- '\\n' + message.content }}\n {%- endif %}\n {%- for tool_call in message.tool_calls %}\n {%- if tool_call.function is defined %}\n {%- set tool_call = tool_call.function %}\n {%- endif %}\n {{- '\\n<tool_call>\\n{\"name\": \"' }}\n {{- tool_call.name }}\n {{- '\", \"arguments\": ' }}\n {{- tool_call.arguments | tojson }}\n {{- '}\\n</tool_call>' }}\n {%- endfor %}\n {{- '<|im_end|>\\n' }}\n {%- elif message.role == \"tool\" %}\n {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n {{- '<|im_start|>user' }}\n {%- endif %}\n {{- '\\n<tool_response>\\n' }}\n {{- message.content }}\n {{- '\\n</tool_response>' }}\n {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n {{- '<|im_end|>\\n' }}\n {%- endif %}\n {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n {{- '<|im_start|>assistant\\n' }}\n{%- else %}\n{{ '<|endoftext|>' }}\n{%- endif %}\n""".lstrip()
    elif isinstance(model, (RND1LM)):
        tokenizer.mask_token = "<|mask|>"
        tokenizer.mask_token_id = 151669
    return tokenizer
