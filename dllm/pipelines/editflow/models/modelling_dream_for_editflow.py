import copy

from dllm.pipelines import dream 
from torch import nn


class DreamModelForEditFlow(dream.DreamModel):
    def __init__(self, config):
        # TODO: time embedding
        super().__init__(config)
        self.sub_logits = copy.deepcopy(self.lm_head)
        self.ins_logits = copy.deepcopy(self.lm_head)
        self.sub_intensity = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        self.ins_intensity = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        self.del_intensity = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softplus())
        self.post_init()

    def forward(self, *args, **kawrgs):
        output = super().forward(*args, **kawrgs)
        h = output["hidden_states"]
        # Position heads
        sub_int = self.sub_intensity(h).squeeze(-1)     # [B, L]
        sub_log = self.sub_logits(h)                    # [B, L, V]
        del_int = self.del_intensity(h).squeeze(-1)     # [B, L]

        # Insertion heads now on token rows (incl. BOS)
        ins_int = self.ins_intensity(h).squeeze(-1)     # [B, L]
        ins_log = self.ins_logits(h)                    # [B, L, V]
        return dict(
            sub_intensity=sub_int,    # [B,L]
            sub_logits=sub_log,       # [B,L,V]
            del_intensity=del_int,    # [B,L]
            ins_intensity=ins_int,    # [B,L]
            ins_logits=ins_log,       # [B,L,V]
        )


if __name__ == "__main__":
    import dllm
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    # Load a config from a local path (either a directory containing config.json, or the file itself)
    config_path = dllm.utils.resolve_with_base_env(
        "Dream-org/Dream-v0-Base-7B", "BASE_MODELS_DIR")
    config = AutoConfig.from_pretrained(config_path)

    # breakpoint()
    torch.set_default_device("cuda") 
    model = DreamModelForEditFlow(config)
    breakpoint()

    # (optional) move to GPU
    import torch
    if torch.cuda.is_available():
        model = model.to("cuda")
