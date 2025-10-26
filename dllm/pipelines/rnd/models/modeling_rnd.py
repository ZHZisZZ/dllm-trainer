# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 model implementation.

This module implements the RND1 architecture with bidirectional attention for
diffusion-based language modeling. Includes support for Mixture of Experts (MoE)
with multiple backend options (HF, vLLM, SGLang, FlashInfer).

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

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeMLP,
    apply_rotary_pos_emb
)
import torch.nn.functional as F


try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as fused_experts_vllm, fused_topk as fused_topk_vllm
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm
except Exception:
    fused_experts_vllm = None
    fused_topk_vllm = None

try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe as sglang_fused_moe
    # from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm # TODO: buggy atm
    from sglang.srt.layers.moe.topk import StandardTopKOutput
except Exception:
    sglang_fused_moe = None
    StandardTopKOutput = None


try:
    import flashinfer.fused_moe as fused_moe
    ## TODO: below needs flashinfer>=0.4.0, but has some bug atm
    # from flashinfer.norm import rmsnorm as flashinfer_rmsnorm
    # class FlashInferRMSNorm(Qwen3MoeRMSNorm):
    #     """Wrapper around FlashInfer RMSNorm to match Qwen3MoeRMSNorm interface"""
    #     def forward(self, hidden_states):
    #         return flashinfer_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
            
except Exception:
    fused_moe = None

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

        if config.moe_backend == "vllm":
            RMSNormClass = VLLMRMSNorm
        else:
            RMSNormClass = Qwen3MoeRMSNorm
        self.q_norm = RMSNormClass(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormClass(self.head_dim, eps=config.rms_norm_eps)

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
            # TODO: modify this to boolean masks
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
        if config.moe_backend == "vllm":
            RMSNormClass = VLLMRMSNorm
        else:
            RMSNormClass = Qwen3MoeRMSNorm
        self.input_layernorm = RMSNormClass(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormClass(config.hidden_size, eps=config.rms_norm_eps)

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
    """RND1 Sparse MoE block with multiple backend support (HF, vLLM, SGLang, FlashInfer)."""

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
        self._w1 = None
        self._w2 = None
        if self.backend == "sglang":
            if sglang_fused_moe is None or StandardTopKOutput is None:
                raise RuntimeError("sglang is not available, cannot use sglang backend")
        elif self.backend == "flashinfer":
            if fused_moe is None:
                raise RuntimeError("flashinfer is not available, cannot use flashinfer backend")
        elif self.backend == "vllm":
            if fused_experts_vllm is None or fused_topk_vllm is None:
                raise RuntimeError("vllm is not available, cannot use vllm backend")

    @torch.no_grad()
    def _initialize_weights(
        self,
        free_experts: bool = True,
        mode: str = "vllm",
        ) -> None:
        logger.info(f"Initializing weights for {mode} backend")
        # Stack directly on device where weights already reside (loaded by HF)
        gate_list: List[torch.Tensor] = []
        up_list: List[torch.Tensor] = []
        down_list: List[torch.Tensor] = []

        # Collect weight references without any device moves
        for expert in self.experts:
            gate_list.append(expert.gate_proj.weight.data)
            up_list.append(expert.up_proj.weight.data)
            down_list.append(expert.down_proj.weight.data)

        gate_w_stacked = torch.stack(gate_list, dim=0).contiguous()
        up_w_stacked = torch.stack(up_list, dim=0).contiguous()
        down_w_stacked = torch.stack(down_list, dim=0).contiguous()

        if mode == "flashinfer":
            w1 = torch.cat([up_w_stacked, gate_w_stacked], dim=1) # FlashInfer expects [up; gate] ordering
        else:
            w1 = torch.cat([gate_w_stacked, up_w_stacked], dim=1)
        w2 = down_w_stacked
        self._w1 = w1
        self._w2 = w2


        if free_experts:
            # Free per-expert modules to reclaim memory
            logger.info(f"Freeing experts for {mode} backend")
            del self.experts
            self.experts = None
        
            
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing and computation."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)

        # Expert routing
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        if self.backend == "vllm":
            routing_weights, selected_experts, *_ = fused_topk_vllm(
                hidden_states=x,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.norm_topk_prob,
            )
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)


        # if self.backend == "hf":
        #     final_hidden_states = torch.zeros(
        #         (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        #     )

        #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        #     expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        #     for expert_idx in expert_hit:
        #         expert_layer = self.experts[expert_idx]
        #         idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        #         current_state = x[top_x]
        #         current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        #         final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        #     out = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        #     return out, router_logits.view(batch_size, sequence_length, -1)
        if self.backend == "hf":
            # Accumulate buffer: [B*S, H]
            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # expert_mask: [E, top_k, tokens]
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).permute(2, 1, 0).contiguous()

            # 顺序遍历所有 experts；即使本 rank 没命中也要进入 forward，避免 ZeRO-3 控制流分歧
            for e in range(self.num_experts):
                expert_layer = self.experts[int(e)]

                # 取出该 expert 命中的 token 索引
                idx, top_x = torch.where(expert_mask[e])        # idx∈[0, top_k), shapes: [n_tok_e]
                current_state = x[top_x]                         # [n_tok_e, H]，n_tok_e 可能为 0
                # if top_x.numel() == 0:
                #     print("0")

                # 空批照样前向；大多数 Linear/MLP 对 0 行输入是 no-op，但会对齐 ZeRO-3 的参数路径
                expert_out = expert_layer(current_state)         # [n_tok_e, H]

                # 路由权重并加权
                w = routing_weights[top_x, idx]                  # [n_tok_e]
                expert_out = expert_out * w.unsqueeze(-1)        # [n_tok_e, H]

                # 累加回全局缓冲；当 n_tok_e=0 时这是合法的空操作
                final_hidden_states.index_add_(0, top_x, expert_out.to(hidden_states.dtype))

            out = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return out, router_logits.view(batch_size, sequence_length, -1)

        elif self.backend == "flashinfer":
            # if self._flashinfer_fc1_weights is None or self._flashinfer_fc2_weights is None:
            #     self._initialize_flashinfer_weights()
            if self._w1 is None or self._w2 is None:
                self._initialize_weights(mode="flashinfer")

            result = fused_moe.cutlass_fused_moe(
                input=x,
                token_selected_experts=selected_experts.to(torch.int),
                token_final_scales=routing_weights.to(torch.float32),
                fc1_expert_weights=self._w1,
                fc2_expert_weights=self._w2,
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
            if self._w1 is None or self._w2 is None:
                self._initialize_weights(mode="sglang")

            topk_output = StandardTopKOutput(
                topk_weights=routing_weights,
                topk_ids=selected_experts,
                router_logits=router_logits,
            )

            out_flat = sglang_fused_moe(
                hidden_states=x,
                w1=self._w1,
                w2=self._w2,
                topk_output=topk_output,
            )
            out = out_flat.view(batch_size, sequence_length, hidden_dim)
            return out, router_logits.view(batch_size, sequence_length, -1)

        elif self.backend == "vllm":
            if self._w1 is None or self._w2 is None:
                self._initialize_weights()

            out_flat = fused_experts_vllm(
                x,
                self._w1,
                self._w2,
                routing_weights,
                selected_experts,
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

        # If configured to use a fused backend, pack fused tensors once after load
        try:
            cfg = getattr(_model, "config", None)
            backend = getattr(cfg, "moe_backend", "hf") if cfg is not None else "hf"
            if backend in ("sglang", "vllm"):
                # Walk decoder layers and initialize fused weights
                model_core = getattr(_model, "model", _model)
                layers = getattr(model_core, "layers", None)
                if isinstance(layers, nn.ModuleList):
                    for layer in layers:
                        mlp = getattr(layer, "mlp", None)
                        if hasattr(mlp, "_initialize_weights"):
                            mlp._initialize_weights(
                                free_experts=True,
                                mode=backend,
                            )
        except Exception as _e:
            logger.warning(f"Backend {backend} weight processing skipped: {_e}")

        return _model


class RND1Model(RND1PreTrainedModel):
    """RND1 transformer model with bidirectional attention for diffusion language modeling."""

    def __init__(self, config: RND1Config):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RND1DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        if config.moe_backend == "vllm":
            RMSNormClass = VLLMRMSNorm
        else:
            RMSNormClass = Qwen3MoeRMSNorm
        self.norm = RMSNormClass(config.hidden_size, eps=config.rms_norm_eps)
        
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

        if isinstance(attention_mask, torch.Tensor):
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

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
