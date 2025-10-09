from typing import Any, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from dllm.utils.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


class LLaDATrainer(transformers.Trainer):

    def __init__(
        self,
        *args,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        **kawrgs,
    ):
        self.scheduler = scheduler or LinearAlphaScheduler()
        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("eps must be in the open interval (0, 1).")
        self.time_epsilon = time_epsilon
        return super().__init__(*args, **kawrgs)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        **kwargs,
    ):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        input_ids, labels = inputs["input_ids"], inputs["labels"]

        b, l = input_ids.shape
        # affine transform: t ∈ [eps, 1)
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1 - self.scheduler(t).unsqueeze(1).repeat(1, l)  # b, 1
        loss_weight = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        masked_indices = masked_indices & (labels != -100)
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).repeat(
            1, l
        )  # b, l

        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )
        outputs = model(input_ids=noised_input_ids)
        logits = outputs.logits

        if not masked_indices.any():
            # return a zero loss that retains graph/device/dtype
            return logits.sum() * 0.0

        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss *= loss_weight[masked_indices]
        loss = (
            torch.sum(token_loss / effective_lengths[masked_indices])
            / input_ids.shape[0]
        )

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    pass
