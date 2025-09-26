from typing import Optional, Text
from datasets import load_dataset, DatasetDict


def load_dataset_opc(dataset_name_or_path: str, name: Optional[Text] = None) -> DatasetDict:
    dataset = load_dataset(dataset_name_or_path, name)

    def map_fn(example):
        return {
            "messages": [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}
            ]
        }

    dataset = dataset.map(map_fn, remove_columns=dataset["train"].column_names, num_proc=4)
    # make train test split
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return dataset


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env
    dataset_name_or_path = resolve_with_base_env("OpenCoder-LLM/opc-sft-stage2", "BASE_DATASETS_DIR")
    dataset = load_dataset_opc(dataset_name_or_path, "educational_instruct")
    breakpoint()
