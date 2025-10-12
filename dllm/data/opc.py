from typing import Optional, Text, List, Dict
from datasets import (
    load_dataset,
    get_dataset_config_names,
    concatenate_datasets,
    DatasetDict,
    Dataset,
)


def _map_to_messages(ds: Dataset) -> Dataset:
    def map_fn(example):
        return {
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    # Remove all original columns after mapping
    remove_cols = ds.column_names
    return ds.map(map_fn, remove_columns=remove_cols, num_proc=4)


def _load_one_config(dataset_name_or_path: str, cfg_name: str) -> Dataset:
    ds = load_dataset(dataset_name_or_path, cfg_name, split="train")
    return _map_to_messages(ds)


def load_dataset_opc(dataset_name_or_path: str, name: str | None = None) -> DatasetDict:
    """
    Load OpenCoder OPC SFT dataset(s) and produce a DatasetDict with a train/test split.
    - If `name` is provided: load that specific config.
    - If `name` is None: load *all* available configs and concatenate them.
    """
    if name is not None:
        train_ds = _load_one_config(dataset_name_or_path, name)
    else:
        # Enumerate and load all configs, then concatenate
        cfgs: list[str] = get_dataset_config_names(dataset_name_or_path)
        if not cfgs:
            raise ValueError(f"No configs found for dataset: {dataset_name_or_path}")
        parts = [_load_one_config(dataset_name_or_path, c) for c in cfgs]
        train_ds = concatenate_datasets(parts)

    # Final split
    ds_dict = train_ds.train_test_split(test_size=0.1, seed=42)
    return DatasetDict(ds_dict)


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "OpenCoder-LLM/opc-sft-stage1", "BASE_DATASETS_DIR"
    )
    # If you want a specific config:
    dataset_edu = load_dataset_opc(dataset_name_or_path, "educational_instruct")
    # Otherwise, all configs concatenated:
    dataset = load_dataset_opc(dataset_name_or_path, None)
    breakpoint()
