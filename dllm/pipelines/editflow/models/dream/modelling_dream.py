import copy
from typing import Optional

from dllm.pipelines import dream
import torch 
from torch import nn


# configuration_dream_editflow.py
from dllm.pipelines.dream import DreamConfig  # or wherever your DreamConfig lives

class EditFlowDreamConfig(DreamConfig):
    model_type = "editflow-dream"  # <- NEW model_type


class EditFlowDreamModel(dream.DreamModel):
    config_class = EditFlowDreamConfig

    def __init__(self, config):
        # TODO: time embedding
        super().__init__(config)
        self.sub_logits = copy.deepcopy(self.lm_head)
        self.ins_logits = copy.deepcopy(self.lm_head)
        # self.sub_rate = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        # self.ins_rate = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        # self.del_rate = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        self.rate_heads = nn.Sequential(nn.Linear(config.hidden_size, 3), nn.Softplus())
        self.post_init()

    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, 
        t: torch.Tensor = None,
        **kwargs
    ):
        # TODO: time embedding
        output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            **kwargs
        )
        h = output["hidden_states"][-1] # final hidden states
        # Position heads
        sub_log = self.sub_logits(h)                    # [B, L, V]
        ins_log = self.ins_logits(h)                    # [B, L, V]
        rates = self.rate_heads(h)               # [B, L, 3]
        sub_rate_hat, del_rate_hat, ins_rate_hat = rates.unbind(-1)    # [B, L], [B, L], [B, L]
        return dict(
            sub_rate_hat=sub_rate_hat,    # [B,L]
            del_rate_hat=del_rate_hat,    # [B,L]
            ins_rate_hat=ins_rate_hat,    # [B,L]

            ins_logits=ins_log,       # [B,L,V]
            sub_logits=sub_log,       # [B,L,V]
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
        "Dream-org/Dream-v0-Base-7B", "BASE_MODELS_DIR")
    config = EditFlowDreamConfig.from_pretrained(config_path)
    breakpoint()
    if hasattr(config, "auto_map"):
        delattr(config, "auto_map")
    if hasattr(config, "architectures"):
        delattr(config, "architectures")

    # breakpoint()
    torch.set_default_device("cuda")
    model = EditFlowDreamModel(config)
    breakpoint()
    model.save_pretrained("models-tmp/editflow")

    auto_model = AutoModel.from_pretrained("models-tmp/editflow")