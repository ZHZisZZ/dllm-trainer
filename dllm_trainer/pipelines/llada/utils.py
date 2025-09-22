from transformers import PreTrainedTokenizer, PreTrainedModel
from dllm_trainer.pipelines.llada.models.modeling_llada import LLaDAModelLM
from dllm_trainer.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM

def postprocess_llada_tokenizer(tokenizer: PreTrainedTokenizer, model: PreTrainedModel):
    if isinstance(model, LLaDAModelLM):
        tokenizer.mask_token = "<|mdm_mask|>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
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
    elif isinstance(model, LLaDAMoEModelLM):
        tokenizer.mask_token = "<|mask|>"
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")
    else:
        raise NotImplementedError
    return tokenizer
