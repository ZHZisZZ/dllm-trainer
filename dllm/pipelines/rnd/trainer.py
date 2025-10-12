from typing import Any

import torch

from dllm.core.trainers import MDLMTrainer


class RNDTrainer(MDLMTrainer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _preprocess_inputs(self, inputs):
        labels = inputs["labels"]
        assert (labels[:, 0] == -100).all()

    def _postprocess_outputs(self, outputs):
        logits = outputs.logits
        outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
