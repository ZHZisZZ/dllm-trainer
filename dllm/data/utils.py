import re

from typing import List, Optional
from datasets import (
    Dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
    load_dataset,
)

from dllm.data.alpaca import load_dataset_alpaca
from dllm.data.opc import load_dataset_opc
from dllm.data.saferlhf import load_dataset_pku_rlhf_sft
from dllm.utils.utils import resolve_with_base_env


def _parse_kv_string(s: str) -> dict:
    return dict(part.split("=", 1) for part in s.split(",") if "=" in part)


def _parse_one_spec(spec: str):
    m = re.search(r"\[(.*?)\]$", spec.strip())
    limits = {}
    if m:
        bracket = m.group(1).strip()
        if bracket:
            for part in bracket.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        f"Invalid split limit entry '{part}' in '{spec}' (expected split:count)."
                    )
                split, count = part.split(":", 1)
                split = split.strip()
                count = count.strip()

                # Accept integers with optional underscores, e.g. 5_000_000
                if not re.fullmatch(r"\d(?:_?\d)*", count):
                    raise ValueError(
                        f"Count must be an integer for '{part}' in '{spec}'. "
                        "Use digits and optional underscores only."
                    )
                limits[split] = int(count.replace("_", ""))

        spec = spec[: m.start()]

    kvs = _parse_kv_string(spec)
    return kvs, limits


def _truncate_split(split_data, n: int):
    if n is None:
        return split_data
    try:
        if hasattr(split_data, "select"):
            # Hugging Face Dataset path
            total = getattr(split_data, "num_rows", None)
            if total is None:
                # some Dataset types expose len(...)
                total = len(split_data)
            idx = list(range(min(n, total)))
            return split_data.select(idx)
    except Exception:
        pass
    try:
        return split_data[:n]
    except Exception:
        # Last resort: iterate
        return type(split_data)(item for i, item in enumerate(split_data) if i < n)


def _truncate_dataset(ds, limits: dict):
    """
    Ensure and return a DatasetDict, truncating splits mentioned in `limits`.
    """
    ds = _ensure_datasetdict(ds)  # normalize first
    out = {}
    for split, data in ds.items():
        n = limits.get(split, None)
        out[split] = _truncate_split(data, n) if n is not None else data
    return DatasetDict(out)


def _concat_splits(a, b):
    """
    Concatenate two split objects (prefer 🤗 datasets).
    """
    if a is b:
        return a
    if a is None:
        return b
    if b is None:
        return a

    # Prefer datasets' concatenate_datasets when both are Datasets
    try:
        from datasets import concatenate_datasets

        if isinstance(a, Dataset) and isinstance(b, Dataset):
            return concatenate_datasets([a, b])
    except Exception:
        pass

    # Fallbacks
    try:
        return a + b
    except Exception:
        pass
    try:
        return type(a)(list(a) + list(b))
    except Exception:
        pass

    raise TypeError(
        f"Cannot concatenate split objects of types {type(a)} and {type(b)}"
    )


def _merge_datasetdicts(d1, d2):
    """
    Merge two DatasetDict-like mappings by concatenating splits present in either.
    Always returns a DatasetDict.
    """
    d1 = _ensure_datasetdict(d1)
    d2 = _ensure_datasetdict(d2)
    all_splits = set(d1.keys()) | set(d2.keys())
    out = {}
    for split in all_splits:
        a = d1.get(split, None)
        b = d2.get(split, None)
        if a is None:
            out[split] = b
        elif b is None:
            out[split] = a
        else:
            out[split] = _concat_splits(a, b)
    return DatasetDict(out)


def _ensure_datasetdict(ds):
    """
    Normalize various loader outputs into a DatasetDict.
    - If loader returns a DatasetDict, return as is.
    - If loader returns a mapping (e.g., dict of splits), wrap into DatasetDict.
    - If loader returns a single Dataset/list/etc., assume it's 'train'.
    """
    if isinstance(ds, DatasetDict):
        return ds
    if isinstance(ds, dict):
        # Try to convert each split value to a Dataset if they aren't already.
        # If they are already Datasets, DatasetDict will accept them directly.
        return DatasetDict(ds)
    # Single split -> assume train
    return DatasetDict({"train": ds})


def _match(name: str, needle) -> bool:
    """
    Returns True if `name` matches any of the provided needles.
    Accepts a single string or a list/tuple of strings.
    Match condition: name endswith(needle) or needle in name.
    """
    if isinstance(needle, (list, tuple)):
        return any(name.endswith(n) or n in name for n in needle)
    return name.endswith(needle) or needle in name


def load_sft_dataset(dataset_args: str):
    """
    Returns:
        datasets.DatasetDict with fields like {"train": Dataset, "test": Dataset, ...}
    Supports:
      - Split limiting via [train:N,test:M,...]
      - Multiple datasets separated by '|', concatenated per split.
      - Unmentioned splits remain full.
    Examples of dataset_args:
      - "dataset_name_or_path=tatsu-lab/alpaca"
      - "dataset_name_or_path=OpenCoder-LLM/opc-sft-stage2,name=educational_instruct"
      - "dataset_name_or_path=tatsu-lab/alpaca[train:5000] | dataset_name_or_path=HuggingFaceH4/ultrachat_200k[train:5000]"
    """
    specs = [p.strip() for p in dataset_args.split("|") if p.strip()]
    all_parts = []

    for raw in specs:
        kvs, limits = _parse_one_spec(raw)
        assert (
            "dataset_name_or_path" in kvs
        ), f"'dataset_name_or_path' missing in spec: {raw}"
        dataset_name_or_path = kvs.pop("dataset_name_or_path")
        dataset_name_or_path = resolve_with_base_env(
            dataset_name_or_path, "BASE_DATASETS_DIR"
        )

        if _match(dataset_name_or_path, "tatsu-lab/alpaca"):
            ds = load_dataset_alpaca(dataset_name_or_path)
        elif _match(dataset_name_or_path, "allenai/tulu-3-sft-mixture"):
            ds = load_dataset(dataset_name_or_path)
            ds = ds["train"].train_test_split(test_size=0.1, seed=42)
        elif _match(dataset_name_or_path, "OpenCoder-LLM/opc-sft-stage1") or _match(
            dataset_name_or_path, "OpenCoder-LLM/opc-sft-stage2"
        ):
            name = kvs.pop("name", None)
            ds = load_dataset_opc(dataset_name_or_path, name)
        elif _match(dataset_name_or_path, "HuggingFaceH4/ultrachat_200k"):
            ds = load_dataset(dataset_name_or_path)
            ds = DatasetDict({"train": ds["train_sft"], "test": ds["test_sft"]})
        elif _match(dataset_name_or_path, "PKU-Alignment/PKU-SafeRLHF"):
            name = kvs.pop("name", None)
            ds = load_dataset_pku_rlhf_sft(dataset_name_or_path, name)
        else:
            raise NotImplementedError(f"Unknown dataset: {dataset_name_or_path}")

        # Normalize to DatasetDict and apply per-split limits
        ds = _ensure_datasetdict(ds)
        ds = _truncate_dataset(ds, limits)
        all_parts.append(ds)

    # If only one part, return as DatasetDict
    if len(all_parts) == 1:
        return _ensure_datasetdict(all_parts[0])

    # Merge all parts into a single DatasetDict
    merged = all_parts[0]
    for part in all_parts[1:]:
        merged = _merge_datasetdicts(merged, part)
    return _ensure_datasetdict(merged)


def _concat_iterable_datasets(parts: List[IterableDataset]) -> IterableDataset:
    """
    Concatenate IterableDatasets sequentially without materialization.
    Preserves streaming nature; supports downstream .take()/.skip()/.shuffle().
    """
    if not parts:
        raise ValueError("No IterableDatasets to concatenate.")
    # Try to reuse features from the first dataset when available
    features = getattr(parts[0], "features", None)

    def _gen():
        for ds in parts:
            for row in ds:
                yield row

    return IterableDataset.from_generator(_gen, features=features)


def _ensure_iterabledatasetdict(obj) -> IterableDatasetDict:
    if isinstance(obj, IterableDatasetDict):
        return obj
    if isinstance(obj, dict):
        return IterableDatasetDict(obj)
    # Single stream -> assume train
    return IterableDatasetDict({"train": obj})


def _merge_iterabledatasetdicts(
    d1: IterableDatasetDict, d2: IterableDatasetDict
) -> IterableDatasetDict:
    """
    Merge by concatenating any overlapping splits (streaming-safe).
    """
    d1 = _ensure_iterabledatasetdict(d1)
    d2 = _ensure_iterabledatasetdict(d2)
    all_splits = set(d1.keys()) | set(d2.keys())
    out = {}
    for split in all_splits:
        a = d1.get(split, None)
        b = d2.get(split, None)
        if a is None:
            out[split] = b
        elif b is None:
            out[split] = a
        else:
            out[split] = _concat_iterable_datasets([a, b])
    return IterableDatasetDict(out)


def _truncate_stream(ds: IterableDataset, n: Optional[int]) -> IterableDataset:
    if n is None:
        return ds
    return ds.take(n)


def load_pt_dataset(dataset_args: str):
    """
    Streaming loader with concatenation across multiple dataset specs (separated by '|').

    Each spec supports:
      - 'dataset_name_or_path=...' (required)
      - Optional [train:N,test:M,...] limits (underscores allowed in integers)

    Current supported datasets (streaming):
      - 'mlfoundations/dclm-baseline-1.0'
        * If both train and test limits are provided, we shuffle the stream with a buffer,
          take (train+test) and split deterministically.
        * If only train is provided, we take that many for 'train' and don't construct 'test'
          unless specified.
        * If only test is provided, we take that many for 'test' from the start (useful for quick eval).
        * If no limits are provided, returns {'train': full_stream}.
    """
    specs = [p.strip() for p in dataset_args.split("|") if p.strip()]
    if not specs:
        raise ValueError("Empty dataset_args for load_pt_dataset.")

    def _load_one_streaming_spec(raw: str) -> IterableDatasetDict:
        kvs, limits = _parse_one_spec(raw)
        assert (
            "dataset_name_or_path" in kvs
        ), f"'dataset_name_or_path' missing in spec: {raw}"
        dataset_name_or_path = kvs.pop("dataset_name_or_path")
        dataset_name_or_path = resolve_with_base_env(
            dataset_name_or_path, "BASE_DATASETS_DIR"
        )

        # ---- Supported streaming datasets ----
        if _match(
            dataset_name_or_path,
            [
                "mlfoundations/dclm-baseline-1.0",
                "OpenCoder-LLM/opc-fineweb-code-corpus",
                "OpenCoder-LLM/opc-fineweb-math-corpus",
            ],
        ):
            # Base stream (single 'train' split on HF hub)
            base = load_dataset(
                dataset_name_or_path,
                split="train",
                streaming=True,
            )
        else:
            raise NotImplementedError(
                f"Streaming dataset not supported: {dataset_name_or_path}"
            )

        # Optional shuffle before slicing to avoid order bias when we split into train/test
        # Tune buffer_size for your I/O memory constraints
        shuffled = base.shuffle(seed=42, buffer_size=10_000)

        n_train = limits.get("train", None)
        n_test = limits.get("test", None)

        if n_train is not None and n_test is not None:
            n_total = n_train + n_test
            head = shuffled.take(n_total)
            # Now split the head deterministically into train/test
            train = head.take(n_train)
            test = head.skip(n_train).take(n_test)
            return IterableDatasetDict({"train": train, "test": test})

        # Only train limit specified
        if n_train is not None and n_test is None:
            train = shuffled.take(n_train)
            return IterableDatasetDict({"train": train})

        # Only test limit specified
        if n_test is not None and n_train is None:
            test = shuffled.take(n_test)
            return IterableDatasetDict({"test": test})

        # No explicit limits → expose full stream as train
        return IterableDatasetDict({"train": shuffled})

    # Load each spec to an IterableDatasetDict and apply *post* truncation per split if extra limits exist.
    # (For sources that already split internally, we respect what _load_one_streaming_spec produced.)
    parts: List[IterableDatasetDict] = []
    for raw in specs:
        ds_part = _load_one_streaming_spec(raw)
        # Apply *additional* per-split truncations if provided in spec (safe no-ops when already exact-sized)
        _, limits = _parse_one_spec(raw)
        truncated = {}
        for split, stream in ds_part.items():
            truncated[split] = _truncate_stream(stream, limits.get(split, None))
        parts.append(IterableDatasetDict(truncated))

    # Merge all parts by concatenating per-split streams
    merged = parts[0]
    for p in parts[1:]:
        merged = _merge_iterabledatasetdicts(merged, p)

    return merged


if __name__ == "__main__":
    # tulu_dataset = load_sft_dataset(
    #     "dataset_name_or_path=allenai/tulu-3-sft-mixture")
    # tulu_datatset_subset = load_sft_dataset(
    #     "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    # )
    # opc_dataset = load_sft_dataset(
    #     "dataset_name_or_path=OpenCoder-LLM/opc-sft-stage2,name=educational_instruct")
    # ultrachat_dataset = load_sft_dataset(
    #     "dataset_name_or_path=HuggingFaceH4/ultrachat_200k")
    # saferlhf_dataset = load_sft_dataset(
    #     "dataset_name_or_path=PKU-Alignment/PKU-SafeRLHF,name=safe")

    # merged_dataset = load_sft_dataset(
    #     "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]|"
    #     "dataset_name_or_path=OpenCoder-LLM/opc-sft-stage2,name=educational_instruct[train:20000,test:2000]"
    # )

    dclm_dataset = load_pt_dataset(
        "dataset_name_or_path=mlfoundations/dclm-baseline-1.0[train:4_500,test:500]"
    )
    dclm_opc_dataset = load_pt_dataset(
        "dataset_name_or_path=mlfoundations/dclm-baseline-1.0[train:4_500,test:500]|"
        "dataset_name_or_path=mlfoundations/dclm-baseline-1.0[train:4_500,test:500]"
    )
    breakpoint()
