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
import datasets


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


def load_peft(model: transformers.PreTrainedModel, training_args: "TrainingArguments") -> transformers.PreTrainedModel:
    if not training_args.lora: return model
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


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError
    return row


def post_process_dataset(
    dataset: datasets.DatasetDict, 
    data_args: "DataArguments"
) -> datasets.DatasetDict:
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
        )
    elif data_args.truncation == "right":
        # do this only if dataset has "prompt_len"
        if "prompt_len" in dataset.column_names["train"]:
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
        )
    else:
        raise NotImplementedError


def clip_row_streaming(row: dict, max_length: int, truncation: str = "right") -> dict:
    """Clip whole sequence OR (if prompt_len present) preserve prompt and clip only the response."""
    if truncation not in {"right", "left"}:
        raise NotImplementedError(f"Unknown truncation: {truncation}")

    def clip(seq):
        return seq[:max_length] if truncation == "right" else seq[-max_length:]

    def clip_preserve_prompt(seq, prompt_len: int):
        prompt = seq[:prompt_len]
        resp   = seq[prompt_len:]
        budget = max(0, max_length - len(prompt))
        resp   = resp[:budget] if truncation == "right" else resp[-budget:]
        return prompt + resp

    prompt_len = row.get("prompt_len", None)
    for k in ("input_ids", "labels", "attention_mask"):
        if k in row and isinstance(row[k], list):
            row[k] = (
                clip_preserve_prompt(row[k], prompt_len)
                if isinstance(prompt_len, int) and prompt_len >= 0
                else clip(row[k])
            )
    return row



def post_process_dataset_streaming(
    dataset: datasets.IterableDatasetDict,
    data_args: "DataArguments",
) -> datasets.IterableDatasetDict:

    def _train_has_prompt_len_streaming(dataset: datasets.IterableDatasetDict) -> bool:
        """Replicates: 'if \"prompt_len\" in dataset.column_names[\"train\"]' for streaming."""
        it = dataset["train"].take(1)
        try:
            ex = next(iter(it))
        except StopIteration:
            return False
        return "prompt_len" in ex

    mode = data_args.truncation
    max_len = data_args.max_length

    if mode == "filter":
        # Keep rows with len(input_ids) <= max_len (emulate .filter with generator map)
        def keep_if_short(row):
            if "input_ids" in row and isinstance(row["input_ids"], list) and len(row["input_ids"]) <= max_len:
                yield row  # keep
            # else: drop (yield nothing)

        return datasets.IterableDatasetDict({name: ds.map(keep_if_short) for name, ds in dataset.items()})

    elif mode == "right":
        ds_out = dataset

        # Do this only if TRAIN split has "prompt_len" (same condition as your non-streaming code)
        if _train_has_prompt_len_streaming(ds_out):
            def keep_if_prompt_fits(row):
                pl = row.get("prompt_len", None)
                if isinstance(pl, int) and pl <= max_len:
                    yield row  # keep
                elif pl is None:
                    # If a row lacks prompt_len but train had it, the non-streaming code would try to access it and fail.
                    # Here we conservatively drop such rows to mirror "requires prompt_len <= max_len".
                    return
                # else: drop

            ds_out = datasets.IterableDatasetDict({name: ds.map(keep_if_prompt_fits) for name, ds in ds_out.items()})

        # Then clip right (same clipping as clip_row)
        def clip_right(row):
            return clip_row(row, max_len, truncation="right")

        return datasets.IterableDatasetDict({name: ds.map(clip_right) for name, ds in ds_out.items()})

    else:
        raise NotImplementedError
