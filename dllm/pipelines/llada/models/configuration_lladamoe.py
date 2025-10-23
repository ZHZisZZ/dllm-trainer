"""
LLaDA MoE configuration
"""
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class LLaDAMoEConfig(PretrainedConfig):
    model_type = "lladamoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=-1,
        hidden_size=-1,
        dense_intermediate_size=-1,
        expert_intermediate_size=-1,
        shared_expert_intermediate_size=-1,
        num_hidden_layers=-1,
        num_attention_heads=-1,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=False,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=-1,
        partial_rotary_factor=-1,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        clip_qkv=None,
        num_experts_per_tok=-1,
        num_experts=-1,
        output_router_logits=False,
        router_aux_loss_coef=0.01,
        norm_topk_prob=None,        
        qk_layernorm=None,
        moe_layer_freq=[],
        moe_router_enable_expert_bias=None,
        moe_router_score_function=None,
        routed_scaling_factor=1,
        router_num_group=-2,
        router_topk_group=-2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.clip_qkv = clip_qkv
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.qk_layernorm = qk_layernorm
        self.moe_layer_freq = moe_layer_freq
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_score_function = moe_router_score_function
        self.partial_rotary_factor = partial_rotary_factor
        self.routed_scaling_factor = routed_scaling_factor
        self.router_num_group = router_num_group
        self.router_topk_group = router_topk_group

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
