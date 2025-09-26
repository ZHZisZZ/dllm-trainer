import os
from contextlib import contextmanager

import pprint
import torch
import accelerate

from dllm.utils.schedulers import BaseScheduler


def looks_like_url_or_scheme(path: str) -> bool:
    # leave URLs, special schemes or cloud URIs untouched
    return any(path.startswith(pfx) for pfx in (
        "http://", "https://", "hf://", "s3://", "gs://", "azure://"
    )) or ("://" in path)


def resolve_with_base_env(path: str, env_name: str) -> str:
    """
    If `env_name` is set and `path` is NOT absolute, NOT a URL/scheme,
    and does not already exist locally, prepend the `env_name` directory.
    Otherwise return `path` unchanged.
    """
    base = os.getenv(env_name, "").strip()
    if not base:
        return path
    if os.path.isabs(path) or looks_like_url_or_scheme(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(base.rstrip("/"), path.lstrip("/"))


def get_num_transfer_tokens(
    mask_index: torch.Tensor, 
    steps: int, 
    scheduler: BaseScheduler, 
    stochastic: bool = False
) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        for t, s, j in zip(range(steps, 0, -1), range(steps-1, -1, -1), range(steps)):
            s /= steps
            t /= steps
            reverse_transfer_prob = 1 - scheduler.reverse_mask_prob(s=s, t=t)
            if not stochastic:
                x = mask_num[i, 0].to(torch.float64) * reverse_transfer_prob
                num_transfer_tokens[i, j] = torch.round(x).to(torch.int64)
            else:
                n = mask_num[i, 0].to(torch.float64)
                num_transfer_tokens[i, j] = torch.distributions.Binomial(n, reverse_transfer_prob).sample().to(torch.int64)
            num_transfer_tokens[i, j] = torch.minimum(num_transfer_tokens[i, j], mask_num[i, 0])
            mask_num[i, 0] -= num_transfer_tokens[i, j]
            if mask_num[i, 0].item() == 0: break
    # Note: because llada is not conditioned on time, this allows us to skip steps with no unmasking (i.e. transfer).
    # Clear all zeros per row (compact) and right-pad with zeros
    # Remove zeros per row, then pad only up to the max length across rows
    rows = []
    max_len = 0
    for i in range(num_transfer_tokens.size(0)):
        nonzero = num_transfer_tokens[i][num_transfer_tokens[i] > 0]
        rows.append(nonzero)
        max_len = max(max_len, nonzero.numel())
    # Pad each row to max_len
    padded_rows = []
    for r in rows:
        if r.numel() < max_len:
            pad = torch.zeros(max_len - r.numel(), dtype=r.dtype, device=r.device)
            r = torch.cat([r, pad])
        padded_rows.append(r)
    return torch.stack(padded_rows, dim=0)


@contextmanager
def init_on(device: str | torch.device, dtype: torch.dtype):
    """
    Temporarily set torch default dtype and default device so that tensors
    created inside the context are allocated on `device` with dtype `dtype`.
    Restores previous settings on exit.
    """
    # Save previous defaults
    prev_dtype = torch.get_default_dtype()
    prev_device = None
    try:
        torch.set_default_dtype(dtype)
        # set_default_device exists in PyTorch >= 2.0
        if hasattr(torch, "set_default_device"):
            # Query current default device if available (optional)
            prev_device = "cpu"
            torch.set_default_device(device)
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")  # revert so DataLoader RNGs stay on CPU


def print_main(*args, **kwargs):
    """
    Print only from the global main process (rank 0 across all nodes).
    Usage: print_main("Hello from main process!")
    """
    if accelerate.PartialState().is_main_process:
        print(*args, **kwargs)


def pprint_main(*args, **kwargs):
    """
    Print (with pprint) only from the global main process (rank 0 across all nodes).
    Usage: print_main("Hello from main process!")
    """
    if accelerate.PartialState().is_main_process:
        pprint.pprint(*args, **kwargs)
