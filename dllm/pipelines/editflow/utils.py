import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Text

import torch
import transformers

# ------------------------------- Collator (x0 source) --------------------------------
# ---------------- Implementations ---------------- #

def sample_x0_empty(*args, **kwargs) -> List[int]:
    """Return BOS-only (i.e. no tokens after BOS)."""
    return []

def sample_x0_with_masks(x1_ids: List[int], tokenizer: Any) -> List[int]:
    """
    Return a run of mask tokens of length ~ Uniform(0.75*|x1_wo_bos|, 1.25*|x1_wo_bos|).
    Ensures at least 1 token so there’s something to edit.
    """
    L = max(0, len(x1_ids) - 1)   # exclude BOS
    if L == 0:
        target_len = 1
    else:
        low = max(1, math.floor(0.75 * L))
        high = max(low, math.ceil(1.25 * L))
        target_len = random.randint(low, high)

    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("tokenizer needs mask_token_id for mask-based sampler")
    return [int(mask_id)] * target_len

# ---------------- Factory ---------------- #

_X0_SAMPLERS: Dict[str, Callable[[List[int], Any], List[int]]] = {
    "sample_x0_empty": sample_x0_empty,
    "sample_x0_with_masks": sample_x0_with_masks,
}

def make_x0_sampler(name: str) -> Callable[[List[int], Any], List[int]]:
    try:
        return _X0_SAMPLERS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown x0 sampler '{name}'. Available: {list(_X0_SAMPLERS)}")

@dataclass
class EditFlowCollator:
    tokenizer: transformers.PreTrainedTokenizer = None
    x0_sampler: Callable | Text | None = sample_x0_with_masks  # can be func OR name

    def _get_sampler(self) -> Callable[[List[int], Any], List[int]]:
        if callable(self.x0_sampler):
            return self.x0_sampler
        if isinstance(self.x0_sampler, str):
            return make_x0_sampler(self.x0_sampler)
        # default fallback
        return sample_x0_with_masks

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not features:
            return {}

        keys = features[0].keys()
        batch = {k: [ex[k] for ex in features] for k in keys}
        batch["x1_ids"] = batch["input_ids"]

        if "labels" not in batch:
            assert all(x1_ids[0] == self.tokenizer.bos_token_id for x1_ids in batch["x1_ids"])
            sampler = self._get_sampler()
            batch["x0_ids"] = [
                x1_ids[:1] + sampler(x1_ids=x1_ids, tokenizer=self.tokenizer)
                for x1_ids in batch["x1_ids"]
            ]
            batch["return_loss"] = True
            return batch
        else:
            raise NotImplementedError


# ------------------------------- Trainer utils --------------------------------

def pad_1d(batch_lists: List[List[int]], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of variable-length integer lists into a tensor [B, Lmax] plus mask [B, Lmax].
    """
    B = len(batch_lists)
    Lmax = max((len(x) for x in batch_lists), default=0)
    out = torch.full((B, Lmax), pad_val, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)
    for b, x in enumerate(batch_lists):
        if len(x) == 0:
            continue
        out[b, :len(x)] = torch.tensor(x, dtype=torch.long)
        mask[b, :len(x)] = True
    return out, mask



if __name__ == "__main__":
    pass
