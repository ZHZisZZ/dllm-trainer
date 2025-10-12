import copy
from typing import Optional

import torch
from torch import nn

from dllm.pipelines import dream


class EditFlowDreamConfig(dream.DreamConfig):
    model_type = "editflow-dream"  # <- NEW model_type


class EditFlowDreamModel(dream.DreamModel):
    config_class = EditFlowDreamConfig
    modules_to_save = {
        "rate_heads",
        "sub_logits",
        "ins_logits",
    }  # fully fintuned even using lora

    def __init__(self, config):
        # TODO: time embedding
        super().__init__(config)
        # lm_head = self.lm_head
        in_lm, out_lm = self.lm_head.in_features, self.lm_head.out_features
        use_bias = self.lm_head.bias is not None
        # Create new, independent heads (no deepcopy)
        self.sub_logits = nn.Linear(in_lm, out_lm, bias=use_bias)
        self.ins_logits = nn.Linear(in_lm, out_lm, bias=use_bias)
        self.rate_heads = nn.Sequential(nn.Linear(config.hidden_size, 3), nn.Softplus())
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        **kwargs
    ):
        # TODO: time embedding
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        h = output["hidden_states"][-1]  # final hidden states
        # Position heads
        sub_log = self.sub_logits(h)  # [B, L, V]
        sub_log = torch.concatenate(
            [torch.zeros_like(sub_log)[:, :1], sub_log[:, :-1]], dim=1
        )  # [B, L, V]
        ins_log = self.ins_logits(h)  # [B, L, V]

        rates = self.rate_heads(h)
        sub_rate_hat, del_rate_hat, ins_rate_hat = rates.unbind(
            -1
        )  # [B, L], [B, L], [B, L]
        sub_rate_hat = torch.concatenate(
            [torch.zeros_like(sub_rate_hat[:, :1]), sub_rate_hat[:, :-1]], dim=1
        )  # [B, L]
        del_rate_hat = torch.concatenate(
            [torch.zeros_like(del_rate_hat[:, :1]), del_rate_hat[:, :-1]], dim=1
        )  # [B, L]
        return dict(
            sub_rate_hat=sub_rate_hat,  # [B,L]
            del_rate_hat=del_rate_hat,  # [B,L]
            ins_rate_hat=ins_rate_hat,  # [B,L]
            ins_logits=ins_log,  # [B,L,V]
            sub_logits=sub_log,  # [B,L,V]
        )


from transformers.models.auto import AutoModel, AutoConfig

# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoConfig.register("editflow-dream", EditFlowDreamConfig)
AutoModel.register(EditFlowDreamConfig, EditFlowDreamModel)


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoConfig, AutoModel

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env(
        "Dream-org/Dream-v0-Base-7B", "BASE_MODELS_DIR"
    )
    config = EditFlowDreamConfig.from_pretrained(config_path)
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    torch.set_default_device("cuda")
    model = EditFlowDreamModel(config)
    model.save_pretrained("models-tmp/editflow-dream")
    auto_model = AutoModel.from_pretrained("models-tmp/editflow-dream")
