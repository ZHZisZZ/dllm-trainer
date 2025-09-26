from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Callable, Text

# @dataclass
# class PassThroughCollator:
#     """
#     A dummy collator that does NOT batch tensors. It returns a dict mapping each
#     field name to a list of per-example values (left exactly as they are).
    
#     Example input (features list of dicts):
#       [{"input_ids": [1,2,3], "labels": 0},
#        {"input_ids": [4,5],   "labels": 1}]
    
#     Output:
#       {
#         "input_ids": [[1,2,3], [4,5]],
#         "labels":    [0, 1]
#       }
#     """
#     include_keys: Optional[Sequence[str]] = None  # keep only these keys if provided
#     drop_keys: Optional[Sequence[str]] = None     # drop these keys if provided
#     fill_missing_with: Any = None                 # value to use if a key is missing in an example

#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
#         if not features:
#             return {}

#         # derive candidate keys from the first example
#         keys = list(features[0].keys())
#         if self.include_keys is not None:
#             keys = [k for k in keys if k in self.include_keys]
#         if self.drop_keys is not None:
#             drop = set(self.drop_keys)
#             keys = [k for k in keys if k not in drop]

#         batch: Dict[str, List[Any]] = {}
#         for k in keys:
#             batch[k] = [
#                 (ex[k] if k in ex else self.fill_missing_with)
#                 for ex in features
#             ]
#         return batch


def sample_x0_empty(*arg, **kwargs):
    """Empty/BOS start (we just use empty here)."""
    # return [[] for _ in range(batch_size)]
    return []

class PassThroughCollator:
    """Return dict[str, list[Any]] without padding/stacking.
       Assumes all examples share the same keys."""
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not features:
            return {}
        keys = features[0].keys()
        return {k: [ex[k] for ex in features] for k in keys}

class EditFlowCollator:
    src_func: Callable | Text = sample_x0_empty

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not features:
            return {}
        keys = features[0].keys()
        batch = {k: [ex[k] for ex in features] for k in keys}
        batch["x1_ids"] = batch["input_ids"]
        # TODO: need to move fast, just assume bos and sample x0 empty
        batch["x0_ids"] = [x1_ids[:1] for x1_ids in batch["x1_ids"]]
        batch["return_loss"] = True
        batch["input_ids"].pop()
        return batch

if __name__ == "__main__":
    pass
