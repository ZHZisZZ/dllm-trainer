from typing import Any, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from dllm.utils.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


def cart_weight(
    masked_indices: torch.Tensor, t: torch.Tensor, p: float = 0.3
) -> torch.Tensor:
    """
    Optimized CART weight computation using matrix operations.

    Args:
        masked_indices (torch.Tensor): (b, l) bool tensor indicating masked positions.
        t (torch.Tensor): (b,) time steps (0-1 sampled uniformly). Not directly used in CART.
        p (float): Parameter of geometric distribution (0 < p <= 1).

    Returns:
        torch.Tensor: (b, l) float tensor of weights.
    """
    b, l = masked_indices.shape
    device = masked_indices.device

    idx = torch.arange(l, device=device)
    dist_matrix = (idx[None, :] - idx[:, None]).abs() - 1
    dist_matrix = torch.clamp(dist_matrix, min=0)  # (l, l)

    geo_matrix = (1 - p) ** dist_matrix * p  # (l, l)

    valid_mask = (~masked_indices).float()  # (b, l), 1 = unmasked
    weights = 0.5 * valid_mask @ geo_matrix.T  # (b, l)
    weights = weights * masked_indices.float()
    return weights


class DreamTrainer(transformers.Trainer):

    def __init__(
        self,
        *args,
        scheduler: Optional[BaseAlphaScheduler] = None,  # CART isn't function of time
        geo_p: float = 0.3,
        **kwargs,
    ):
        self.scheduler = scheduler or LinearAlphaScheduler()
        self.geo_p = geo_p
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: Union[transformers.PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape

        # 1. sample timesteps
        t = torch.rand(b, device=input_ids.device)  # (b,)
        p_mask = 1 - self.scheduler(t).unsqueeze(1).repeat(1, l)  # (b, l)

        # 2. apply masking
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        masked_indices = masked_indices & (labels != -100)
        # Dream cannot unmask when the mask is the first token.
        masked_indices[:, 0] = False
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # 3. forward
        outputs = model(noised_input_ids, attention_mask)
        logits = outputs.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        if not masked_indices.any():
            # return a zero loss that retains graph/device/dtype
            return logits.sum() * 0.0

        # 4. compute CART weights
        loss_weights = cart_weight(masked_indices, t, p=self.geo_p)

        # 5. compute weighted cross entropy
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # normalization
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).repeat(1, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    pass
