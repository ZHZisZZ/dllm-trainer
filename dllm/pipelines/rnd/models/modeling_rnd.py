
# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 model implementation.

This module implements the RND1 architecture with bidirectional attention for
diffusion-based language modeling. Includes support for Mixture of Experts (MoE)
with multiple backend options (HF, FlashInfer, SGLang).

Based on the Qwen3Moe architecture:
https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List, Union

import torch
from torch import nn

from transformers.utils import logging
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MaskedLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig

from .configuration_rnd import RND1Config
from .generation_utils import RND1GenerationMixin
from .generation_config import RND1GenerationConfig

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
    apply_rotary_pos_emb
)
import torch.nn.functional as F

try:
    import flashinfer.fused_moe as fused_moe
except Exception:
    fused_moe = None

try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as sglang_fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput
except Exception:
    sglang_fused_moe = None
    StandardTopKOutput = None

logger = logging.get_logger(__name__)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match query heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RND1Attention(nn.Module):
    """RND1 attention layer with bidirectional attention for diffusion modeling."""

    def __init__(self, config: RND1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = getattr(config, "sliding_window", None)

        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Union[Cache, Tuple[torch.Tensor, torch.Tensor]]]]:

        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        use_sdpa = (getattr(self.config, "_attn_implementation", "eager") == "sdpa")

        if use_sdpa:
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                if attention_mask.dtype not in [torch.bool, torch.float32, torch.float16, torch.bfloat16]:
                    attention_mask = attention_mask.to(dtype=query_states.dtype)
            
            assert not self.is_causal, f"Attention layer {self.layer_idx} is causal"
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(bsz, q_len, self.num_heads * self.head_dim)
            attn_out = self.o_proj(attn_out)
            return attn_out, None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_out = torch.matmul(attn_weights, value_states)
        attn_out = attn_out.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), -1)
        attn_out = self.o_proj(attn_out)

        return attn_out, None


class RND1DecoderLayer(nn.Module):
    """RND1 decoder layer with bidirectional attention for diffusion language modeling."""

    def __init__(self, config: RND1Config, layer_idx: int):
        super().__init__()
        self.self_attn = RND1Attention(config, layer_idx)
        self.mlp = RND1SparseMoeBlock(config)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        replace_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ff_out = self.mlp(hidden_states)
        if isinstance(ff_out, tuple):
            ff_out = ff_out[0]
        hidden_states = residual + ff_out

        return hidden_states, attn_weights


class RND1SparseMoeBlock(nn.Module):
    """RND1 Sparse MoE block with multiple backend support (HF, FlashInfer, SGLang)."""

    def __init__(self, config: RND1Config):
        super().__init__()
        self.config = config
        self.backend = getattr(config, "moe_backend", "hf")
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MoeMLP(config, intermediate_size=self.intermediate_size) for _ in range(self.num_experts)]
        )

        # Cached weight tensors for optimized backends
        self._flashinfer_fc1_weights = None
        self._flashinfer_fc2_weights = None
        self._sglang_w1 = None
        self._sglang_w2 = None
        if self.backend == "sglang":
            if sglang_fused_moe is None or StandardTopKOutput is None:
                raise RuntimeError("sglang is not available, cannot use sglang backend")
        elif self.backend == "flashinfer":
            if fused_moe is None:
                raise RuntimeError("flashinfer is not available, cannot use flashinfer backend")

    def _initialize_flashinfer_weights(self):
        """Initialize FlashInfer-compatible weight format."""
        fc1_list = []
        fc2_list = []

        for expert in self.experts:
            gate_w = expert.gate_proj.weight  # [I, H]
            up_w = expert.up_proj.weight      # [I, H]
            down_w = expert.down_proj.weight  # [H, I]
            # FlashInfer expects [up; gate] ordering
            fc1_list.append(torch.cat([up_w, gate_w], dim=0))  # [2I, H]
            fc2_list.append(down_w)  # [H, I]

        self._flashinfer_fc1_weights = torch.stack(fc1_list, dim=0).contiguous()
        self._flashinfer_fc2_weights = torch.stack(fc2_list, dim=0).contiguous()

    def _initialize_sglang_weights(self):
        """Initialize SGLang-compatible weight format."""
        w1_list = []
        w2_list = []

        for expert in self.experts:
            gate_w = expert.gate_proj.weight  # [I, H]
            up_w = expert.up_proj.weight      # [I, H]
            down_w = expert.down_proj.weight  # [H, I]
            w1 = torch.cat([gate_w, up_w], dim=0)  # [2I, H]
            w1_list.append(w1)
            w2_list.append(down_w)

        self._sglang_w1 = torch.stack(w1_list, dim=0).contiguous()
        self._sglang_w2 = torch.stack(w2_list, dim=0).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing and computation."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        # Expert routing
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        if self.backend == "hf":
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
                current_state = x[top_x]
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            out = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return out, router_logits.view(batch_size, sequence_length, -1)

        elif self.backend == "flashinfer":
            if self._flashinfer_fc1_weights is None or self._flashinfer_fc2_weights is None:
                self._initialize_flashinfer_weights()

            result = fused_moe.cutlass_fused_moe(
                input=x,
                token_selected_experts=selected_experts.to(torch.int),
                token_final_scales=routing_weights.to(torch.float32),
                fc1_expert_weights=self._flashinfer_fc1_weights,
                fc2_expert_weights=self._flashinfer_fc2_weights,
                output_dtype=x.dtype,
                quant_scales=None,
            )
            if isinstance(result, (list, tuple)):
                out_flat = result[0]
            else:
                out_flat = result
            out = out_flat.view(batch_size, sequence_length, hidden_dim)
            return out, router_logits.view(batch_size, sequence_length, -1)

        elif self.backend == "sglang":
            if self._sglang_w1 is None or self._sglang_w2 is None:
                self._initialize_sglang_weights()

            topk_output = StandardTopKOutput(
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                router_logits=router_logits,
            )

            out_flat = sglang_fused_moe(
                hidden_states=x,
                w1=self._sglang_w1,
                w2=self._sglang_w2,
                topk_output=topk_output,
            )
            out = out_flat.view(batch_size, sequence_length, hidden_dim)
            return out, router_logits.view(batch_size, sequence_length, -1)

        else:
            raise ValueError(f"Invalid backend: {self.backend}")


class RND1PreTrainedModel(PreTrainedModel):
    """Base class for RND1 models with weight initialization and loading support."""
    config_class = RND1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RND1DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        """Load pretrained model with generation config."""
        _model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        
        resume_download = kwargs.get("resume_download", None)
        proxies = kwargs.get("proxies", None)
        subfolder = kwargs.get("subfolder", "")
        from_auto_class = kwargs.get("_from_auto", False)
        from_pipeline = kwargs.get("_from_pipeline", None)
        
        _model.generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
        )
            
        return _model


class RND1Model(RND1PreTrainedModel):
    """RND1 transformer model with bidirectional attention for diffusion language modeling."""

    def __init__(self, config: RND1Config):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RND1DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        """Forward pass through the RND1 model."""

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            router_logits=None,
        )


class RND1LM(RND1PreTrainedModel, RND1GenerationMixin):
    """Radical Numerics Diffusion Language Model with bidirectional attention."""

    def __init__(self, config: RND1Config):
        super().__init__(config)
        self.model = RND1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings layer."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set the input embeddings layer."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Get the output embeddings layer (lm_head)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings layer (lm_head)."""
        self.lm_head = new_embeddings

    @classmethod
    def can_generate(cls) -> bool:
        """Indicates this model can generate text."""
        return True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MaskedLMOutput:
        """Forward pass with optional loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )


from transformers import AutoModel
# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(RND1Config, RND1LM)
