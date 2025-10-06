import math
import random
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Callable, Text

import torch
import transformers

import dllm

# ------------------------------- Collator (x0 source) --------------------------------
# ---------------- Utilities ---------------- #

def _special_id_set(tokenizer: Any) -> set:
    s = set()
    for attr in [
        "all_special_ids", "bos_token_id", "eos_token_id", "pad_token_id",
        "unk_token_id", "cls_token_id", "sep_token_id", "mask_token_id"
    ]:
        v = getattr(tokenizer, attr, None)
        if v is None: 
            continue
        if isinstance(v, (list, tuple)):
            s.update(int(x) for x in v if x is not None)
        else:
            s.add(int(v))
    return s

def _rand_vocab_token(tokenizer: Any, *, exclude: set) -> int:
    # Uniform draw over vocab \ exclude
    V = getattr(tokenizer, "vocab_size", None)
    if V is None:
        # Fallback for some tokenizers
        V = len(getattr(tokenizer, "get_vocab")())
    while True:
        tok = random.randint(0, V - 1)
        if tok not in exclude:
            return tok

# ---------------- Implementations ---------------- #

def sample_x0_empty(*args, **kwargs) -> List[int]:
    """Return BOS-only (i.e. no tokens after BOS)."""
    return []

def sample_x0_masks(tokenizer: Any, *args, target_len=128, **kwargs) -> List[int]:
    """
    Return a run of mask tokens of length `target_len`.
    """
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise ValueError("tokenizer needs mask_token_id for mask-based sampler")
    return [int(mask_id)] * target_len

def sample_x0_noisy(
    x1_ids: List[int],
    tokenizer: Any,
    # Per-token action probs (renormalized; p_mask ignored if tokenizer has no mask id)
    p_del: float = 0.20,   # delete (teaches later INSERT)
    p_sub: float = 0.15,   # substitute with a different vocab token (teaches SUB)
    p_mask: float = 0.15,  # replace with [MASK]
    p_keep: float = 0.50,  # keep as-is
    # Independent chance to insert a distractor token *after* each processed position:
    p_ins_after: float = 0.05,
    # ensure_min_len: bool = True,
    target_len: int = 128,
) -> List[int]:
    """
    Build x0 by applying Delete/Substitute/Mask/Keep to the provided sequence, then
    right-truncate or pad to exactly `target_len`. BOS is *not* expected here.
    """
    if not x1_ids:
        x1_ids = []

    specials = _special_id_set(tokenizer)
    mask_id = getattr(tokenizer, "mask_token_id", None)

    # Effective probabilities (drop p_mask if no mask id; renormalize)
    p_del = max(0.0, p_del)
    p_sub = max(0.0, p_sub)
    p_keep = max(0.0, p_keep)
    p_mask_eff = max(0.0, p_mask) if mask_id is not None else 0.0
    s = p_del + p_sub + p_mask_eff + p_keep
    if s <= 0.0:
        # robust fallback
        p_del, p_sub, p_mask_eff, p_keep = (0.1, 0.1, 0.2 if mask_id is not None else 0.0, 0.6)
        s = p_del + p_sub + p_mask_eff + p_keep
    p_del, p_sub, p_mask_eff, p_keep = (p_del/s, p_sub/s, p_mask_eff/s, p_keep/s)

    out: List[int] = []
    for tok in x1_ids:
        r = random.random()
        if r < p_del:
            # DELETE
            pass
        elif r < p_del + p_sub:
            # SUBSTITUTE (avoid specials and original token)
            exclude = specials | {int(tok)}
            sub_tok = _rand_vocab_token(tokenizer, exclude=exclude)
            out.append(int(sub_tok))
        elif r < p_del + p_sub + p_mask_eff:
            # MASK
            out.append(int(mask_id))  # type: ignore[arg-type]
        else:
            # KEEP
            out.append(int(tok))

        # Optional distractor INSERT after this position
        if random.random() < p_ins_after:
            ins_tok = _rand_vocab_token(tokenizer, exclude=specials)
            out.append(int(ins_tok))

    # Enforce exact target length:
    # 1) truncate on the right if too long
    if len(out) > target_len:
        out = out[:target_len]
    # 2) pad on the right if too short (prefer [MASK], else random non-special)
    elif len(out) < target_len:
        pad_needed = target_len - len(out)
        if mask_id is not None:
            out.extend([int(mask_id)] * pad_needed)
        else:
            for _ in range(pad_needed):
                out.append(int(_rand_vocab_token(tokenizer, exclude=specials)))

    return out

def sample_x0_mixture(
    x1_ids: List[int],
    tokenizer: Any,
    *args,
    w_empty: float = 0.20,          # teaches INSERT beyond prompt
    w_noisy: float = 0.20,          # teaches SUB + DEL + KEEP
    w_masks: float = 0.60,          # optional mask-run variety
    **kwargs,
    # You can pass through knobs for noisy/mask variants by editing defaults here
) -> List[int]:
    """
    Sample x0 from a mixture:
      - prompt-only (empty tail): insert-heavy supervision
      - noisy-tokens: substitution/deletion supervision
      - masks: optional variety
    """
    ws = [max(0.0, w_empty), max(0.0, w_noisy), max(0.0, w_masks)]
    s = sum(ws)
    # if s == 0: ws = [0.3, 0.5, 0.2]
    # else: ws = [w/s for w in ws]
    ws = [w/s for w in ws]
    r = random.random()
    if r < ws[0]:
        return sample_x0_empty(x1_ids=x1_ids, tokenizer=tokenizer)
    elif r < ws[0] + ws[1]:
        return sample_x0_noisy(x1_ids=x1_ids, tokenizer=tokenizer)
    else:
        return sample_x0_masks(x1_ids=x1_ids, tokenizer=tokenizer)

# ---------------- Factory ---------------- #

_X0_SAMPLERS: Dict[str, Callable[[List[int], Any], List[int]]] = {
    "sample_x0_empty": sample_x0_empty,
    "sample_x0_masks": sample_x0_masks,
    "sample_x0_noisy": sample_x0_noisy,
    "sample_x0_mixture": sample_x0_mixture,
}

def make_x0_sampler(name: str) -> Callable[[List[int], Any], List[int]]:
    try:
        return _X0_SAMPLERS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown x0 sampler '{name}'. Available: {list(_X0_SAMPLERS)}")

@dataclass
class EditFlowCollator:
    tokenizer: transformers.PreTrainedTokenizer = None
    x0_sampler: Callable | Text | None = sample_x0_masks  # can be func OR name

    def _get_sampler(self) -> Callable[[List[int], Any], List[int]]:
        if callable(self.x0_sampler):
            return self.x0_sampler
        if isinstance(self.x0_sampler, str):
            return make_x0_sampler(self.x0_sampler)
        # default fallback
        return sample_x0_masks

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not features:
            return {}

        keys = features[0].keys()
        batch = {k: [ex[k] for ex in features] for k in keys}
        batch["x1_ids"] = batch["input_ids"]

        sampler = self._get_sampler()

        if "prompt_len" not in batch:
            assert all(x1_ids[0] == self.tokenizer.bos_token_id for x1_ids in batch["x1_ids"])
            batch["x0_ids"] = [
                x1_ids[:1] + sampler(x1_ids=x1_ids[1:], tokenizer=self.tokenizer)
                for x1_ids in batch["x1_ids"]
            ]
        else:
            batch["x0_ids"] = [
                x1_ids[:prompt_len] + sampler(x1_ids=x1_ids[prompt_len:], tokenizer=self.tokenizer)
                for x1_ids, prompt_len in zip(batch["x1_ids"], batch["prompt_len"])
            ]
    
        batch["return_loss"] = True
        return batch


# ------------------------------- Trainer utils --------------------------------

def pad_1d(batch_lists: List[List[int]], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of variable-length integer lists into:
      - out: tensor of shape [B, Lmax] with padding value `pad_val`
      - mask: tensor of shape [B, Lmax] with 1 for real tokens and 0 for padding (int mask)
    """
    B = len(batch_lists)
    Lmax = max((len(x) for x in batch_lists), default=0)
    out = torch.full((B, Lmax), pad_val, dtype=torch.long)
    mask = torch.zeros((B, Lmax), dtype=torch.long)  # 0/1 mask (int)

    for b, x in enumerate(batch_lists):
        if not x:
            continue
        L = len(x)
        out[b, :L] = torch.tensor(x, dtype=torch.long)
        mask[b, :L] = 1  # mark valid positions with 1

    return out, mask


# def init_editflow_from_src(ef_model, src_model, lm_head_key="lm_head", verbose=True):
#     """
#     Initialize an EditFlowModel (ef_model) from a pretrained source model.

#     - Copies all matching backbone parameters.
#     - Duplicates source lm_head -> ef_model.sub_logits and ef_model.ins_logits.
#     - Leaves new rate heads (sub_rate/ins_rate/del_rate) randomly initialized.
#     - Returns (missing_keys, unexpected_keys) from load_state_dict(strict=False).

#     Args:
#         ef_model:    EditFlowModel instance (target).
#         src_model:   Source model instance (the pretrained backbone).
#         lm_head_key: Base key name for the source model's LM head (default: "lm_head").
#         verbose:     If True, prints a short load report.

#     Example:
#         src = SomeBackboneModel.from_pretrained(path)
#         ef  = EditFlowModel.from_config(cfg)
#         init_editflow_from_src(ef, src)
#     """
#     src_sd = src_model.state_dict()
#     tgt_sd = ef_model.state_dict()
#     new_sd = OrderedDict()

#     # 1) copy matching tensors (same key & shape)
#     for k, v in src_sd.items():
#         if k in tgt_sd and tgt_sd[k].shape == v.shape:
#             new_sd[k] = v

#     # 2) duplicate lm_head -> sub_logits & ins_logits (weight + optional bias)
#     lm_w = f"{lm_head_key}.weight"
#     lm_b = f"{lm_head_key}.bias"

#     if lm_w in src_sd:
#         if "sub_logits.weight" in tgt_sd:
#             new_sd["sub_logits.weight"] = src_sd[lm_w]
#         if "ins_logits.weight" in tgt_sd:
#             new_sd["ins_logits.weight"] = src_sd[lm_w]

#     if lm_b in src_sd:
#         if "sub_logits.bias" in tgt_sd:
#             new_sd["sub_logits.bias"] = src_sd[lm_b]
#         if "ins_logits.bias" in tgt_sd:
#             new_sd["ins_logits.bias"] = src_sd[lm_b]

#     # 3) load non-strictly so new heads can stay randomly initialized
#     missing, unexpected = ef_model.load_state_dict(new_sd, strict=False)

#     if verbose:
#         dllm.utils.print_main(f"[EditFlow init] Copied {len(new_sd)} tensors from Src Model.")
#         if missing:
#             dllm.utils.print_main("  Missing (expected for new rate heads, etc.):")
#             for k in missing:
#                 dllm.utils.print_main("   -", k)
#         if unexpected:
#             dllm.utils.print_main("  Unexpected (check key names):")
#             for k in unexpected:
#                 dllm.utils.print_main("   -", k)

#     return missing, unexpected
def init_editflow_from_src(ef_model, src_model, lm_head_key: str = "lm_head", verbose: bool = True):
    """
    Initialize an EditFlowModel (ef_model) from a pretrained source model.

    If DeepSpeed ZeRO-3 is enabled (detected via HF's `is_deepspeed_zero3_enabled()`),
    this function temporarily gathers full parameters for both models on rank 0,
    performs the copy there, and then returns to sharded mode automatically.
    Otherwise it behaves like a normal CPU/GPU single-process copy.

    Returns (missing_keys, unexpected_keys) from load_state_dict(strict=False).
    """
    import deepspeed
    from transformers.integrations import is_deepspeed_zero3_enabled
    dist_ok = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if dist_ok else 0

    def _copy_once():
        src_sd = src_model.state_dict()
        tgt_sd = ef_model.state_dict()
        new_sd = OrderedDict()

        # 1) copy matching backbone tensors
        for k, v in src_sd.items():
            if k in tgt_sd and tgt_sd[k].shape == v.shape:
                new_sd[k] = v

        # 2) duplicate lm_head -> sub_logits & ins_logits (weight + optional bias)
        lm_w = f"{lm_head_key}.weight"
        lm_b = f"{lm_head_key}.bias"

        if lm_w in src_sd:
            if "sub_logits.weight" in tgt_sd:
                new_sd["sub_logits.weight"] = src_sd[lm_w]
            if "ins_logits.weight" in tgt_sd:
                new_sd["ins_logits.weight"] = src_sd[lm_w]
        if lm_b in src_sd:
            if "sub_logits.bias" in tgt_sd:
                new_sd["sub_logits.bias"] = src_sd[lm_b]
            if "ins_logits.bias" in tgt_sd:
                new_sd["ins_logits.bias"] = src_sd[lm_b]

        # 3) non-strict load so new rate heads remain randomly initialized
        missing, unexpected = ef_model.load_state_dict(new_sd, strict=False)
        return new_sd, missing, unexpected

    if is_deepspeed_zero3_enabled():
        # All ranks enter/exit together; only rank 0 materializes full tensors.
        params = list(ef_model.parameters()) + list(src_model.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
            if rank == 0:
                new_sd, missing, unexpected = _copy_once()
            else:
                new_sd, missing, unexpected = OrderedDict(), [], []

        if dist_ok:
            torch.distributed.barrier()

        if verbose and rank == 0:
            _p = getattr(globals().get("dllm", None), "utils", None)
            printer = getattr(_p, "print_main", print) if _p else print
            printer(f"[EditFlow init][ZeRO-3] Copied {len(new_sd)} tensors from Src Model.")
            if missing:
                printer("  Missing (expected for new rate heads, etc.):")
                for k in missing: printer("   -", k)
            if unexpected:
                printer("  Unexpected (check key names):")
                for k in unexpected: printer("   -", k)
        return missing, unexpected

    # --- Non-ZeRO (or DS not present) path ---
    new_sd, missing, unexpected = _copy_once()
    if verbose:
        _p = getattr(globals().get("dllm", None), "utils", None)
        printer = getattr(_p, "print_main", print) if _p else print
        printer(f"[EditFlow init] Copied {len(new_sd)} tensors from Src Model.")
        if missing:
            printer("  Missing (expected for new rate heads, etc.):")
            for k in missing: printer("   -", k)
        if unexpected:
            printer("  Unexpected (check key names):")
            for k in unexpected: printer("   -", k)
    return missing, unexpected


if __name__ == "__main__":
    pass
