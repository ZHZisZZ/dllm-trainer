import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dllm.utils.configs import ModelArguments, DataArguments, TrainingArguments

import pprint
import torch
import peft
import accelerate
import transformers


def resolve_with_base_env(path: str, env_name: str) -> str:
    """
    If `env_name` is set and `path` is NOT absolute, NOT a URL/scheme,
    and does not already exist locally, prepend the `env_name` directory.

    If the resulting path does not exist, return the base environment directory instead.
    Otherwise return `path` unchanged.
    """
    base = os.getenv(env_name, "").strip()
    if not base:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path

    candidate = os.path.join(base.rstrip("/"), path.lstrip("/"))
    if os.path.exists(candidate):
        return candidate
    else:
        return base


@contextmanager
def init_device_context_manager(device: str | torch.device | None = None):
    """
    Temporarily set torch default dtype and default device so that tensors
    created inside the context are allocated on `device` with dtype `dtype`.
    Restores previous settings on exit.
    """
    if transformers.integrations.is_deepspeed_zero3_enabled():
        yield
        return

    # Resolve device
    if device is None:
        try:
            from accelerate import PartialState

            idx = PartialState().local_process_index
        except Exception:
            idx = 0
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}"

    try:
        torch.set_default_device(device)
        yield
    finally:
        torch.set_default_device("cpu")


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


def load_peft(
    model: transformers.PreTrainedModel, training_args: "TrainingArguments"
) -> transformers.PreTrainedModel:
    if not training_args.lora:
        return model
    peft_config = peft.LoraConfig(
        r=training_args.r,
        target_modules=training_args.target_modules,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.bias,
        modules_to_save=getattr(model, "modules_to_save", None),
    )
    model = peft.get_peft_model(model, peft_config)
    if accelerate.PartialState().is_main_process:
        print(model)
        model.print_trainable_parameters()
    return model


def print_args_main(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
):
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


def disable_caching_allocator_warmup():
    try:
        from transformers import modeling_utils as _mu

        def _noop(*args, **kwargs):
            return

        _mu.caching_allocator_warmup = _noop
    except Exception:
        pass


def disable_dataset_progress_bar_except_main():
    # state = accelerate.PartialState()  # figures out your rank/world automatically
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar

    if accelerate.PartialState().is_main_process:
        enable_progress_bar()
    else:
        disable_progress_bar()


def initial_training_setup(training_args: "TrainingArguments"):
    transformers.set_seed(training_args.seed)
    disable_caching_allocator_warmup()
    disable_dataset_progress_bar_except_main()


def parse_spec(spec: str):
    """
    Parse a general 'name[a:b,c:d]' or 'a=b,c=d' style specification.

    Supports:
      - Bare name, e.g. "foo/bar"
      - Optional bracket suffix with comma-separated entries:
          key:value or key:int_value (underscores allowed)
      - Optional "key=value" pairs outside the bracket.

    Returns:
      name: str or None
      kv_dict: dict of key/value pairs (all combined)
    """

    def _parse_kv_string(s: str) -> dict:
        """Parse comma-separated key=value pairs, e.g. 'a=1,b=2'."""
        return dict(part.split("=", 1) for part in s.split(",") if "=" in part)

    s = spec.strip()

    # Extract bracket content if present
    m = re.search(r"\[(.*?)\]$", s)
    bracket_kvs = {}
    numeric_kvs = {}
    if m:
        bracket = m.group(1).strip()
        if bracket:
            for part in bracket.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid entry '{part}' in '{spec}' (expected key:value)."
                    )
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Integers (with optional underscores)
                if re.fullmatch(r"\d(?:_?\d)*", value):
                    numeric_kvs[key] = int(value.replace("_", ""))
                else:
                    bracket_kvs[key] = value

        # Remove the bracket suffix from the working string
        s = s[: m.start()].rstrip()

    # Determine name (if any) and parse outer kvs (if any)
    name = None
    if "=" in s:
        kv_dict = dict(_parse_kv_string(s))
    else:
        kv_dict = {}
        if s:
            name = s  # could represent a dataset, resource, or identifier

    # Merge: bracket options and numeric keys last
    kv_dict.update(bracket_kvs)
    kv_dict.update(numeric_kvs)

    return name, kv_dict
