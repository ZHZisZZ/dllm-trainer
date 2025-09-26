from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Text

import torch

# ------------------------------- Collator (x0 source) --------------------------------

def sample_x0_empty(*arg, **kwargs):
    """Empty/BOS start (we just use empty here)."""
    # return [[] for _ in range(batch_size)]
    return []

@dataclass
class EditFlowCollator:
    tokenizer: Any = None
    src_func: Callable | Text | None = sample_x0_empty

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not features:
            return {}
        keys = features[0].keys()
        batch = {k: [ex[k] for ex in features] for k in keys}
        batch["x1_ids"] = batch["input_ids"]
        if not "labels" in batch:
            # TODO: need to move fast, just assume bos and sample x0 empty
            assert all(x1_ids[0] == self.tokenizer.bos_token_id for x1_ids in batch["x1_ids"])
            batch["x0_ids"] = [x1_ids[:1] + self.src_func(x1_ids) for x1_ids in batch["x1_ids"]]
            batch["return_loss"] = True
            # batch["input_ids"].pop()
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
