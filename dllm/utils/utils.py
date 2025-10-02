import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING
if TYPE_CHECKING: from dllm.utils.configs import ModelArguments, DataArguments, TrainingArguments

import pprint
import torch
import peft
import accelerate
import transformers


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


def load_peft(model: transformers.PreTrainedModel, peft_args: "ModelArguments") -> transformers.PreTrainedModel:
    if not peft_args.lora: return model
    peft_config = peft.LoraConfig(
        r=peft_args.r,
        target_modules=peft_args.target_modules,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias=peft_args.bias,
    )
    model = peft.get_peft_model(model, peft_config)
    if accelerate.PartialState().is_main_process:
        model.print_trainable_parameters()
    return model


def print_args_main(model_args: "ModelArguments", data_args: "DataArguments", training_args: "TrainingArguments"):
    print_main("\n===== Parsed arguments =====")
    for name, args in [
        ("model_args", model_args),
        ("data_args", data_args),
        ("training_args", training_args),
    ]:
        d = asdict(args)
        # keep it tiny: just show first few entries
        short = {k: d[k] for k in list(d)}  # adjust number as you like
        print_main(f"{name}:")
        pprint_main(short, width=100, compact=True)
    print_main("============================\n")
