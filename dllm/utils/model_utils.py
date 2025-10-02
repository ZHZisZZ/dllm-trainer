import torch
import accelerate
import transformers

from dllm.utils.configs import ModelArguments, TrainingArguments


def get_model(
    model_args: ModelArguments, 
    training_args: TrainingArguments | None = None
) -> transformers.PreTrainedModel:
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

    torch_dtype = getattr(model_args, "torch_dtype", "bfloat16")
    torch_dtype = dtype_map.get(str(torch_dtype).lower())

    if not training_args:
        return transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch_dtype, device_map="auto", 
        )

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


def get_tokenizer(
    model_args: ModelArguments, 
    model: transformers.PreTrainedModel | None = None
) -> transformers.PreTrainedTokenizer:
    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM
    from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM
    from dllm.pipelines.dream.models.modeling_dream import DreamModel
    from dllm.pipelines.editflow.models.dream.modelling_dream import EditFlowDreamModel
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
    )
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    if not model: return tokenizer
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
    if isinstance(model, (DreamModel, EditFlowDreamModel)):
        # TODO: add postprocessing to fix chat template
        pass
    return tokenizer
