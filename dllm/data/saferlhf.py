from typing import Literal
from datasets import load_dataset, DatasetDict


def _keep_harmful_prompts(row):
    # Keep prompts that have at least one unsafe response
    return not (bool(row["is_response_0_safe"]) and bool(row["is_response_1_safe"]))


def load_dataset_pku_rlhf_sft(
    dataset_name_or_path: str,
    name: Literal["safe", "unsafe"] = "safe",
) -> DatasetDict:
    """
    Return a dataset aligned with opc.py:
    - Each sample has only one field: `messages = [
        {"role": "user", "content": <prompt>},
        {"role": "assistant", "content": <response>}
      ]`
    - Perform a 9:1 train/test split
    """
    if name not in ("safe", "unsafe"):
        raise NotImplementedError(f"Unsupported name: {name}")

    dataset = load_dataset(dataset_name_or_path)
    # Keep only prompts with at least one unsafe response (same as original logic)
    dataset = dataset.filter(_keep_harmful_prompts, num_proc=4)

    def to_messages(batch):
        msgs = []
        for i in range(len(batch["prompt"])):
            p = batch["prompt"][i]

            r0_safe = bool(batch["is_response_0_safe"][i])
            r1_safe = bool(batch["is_response_1_safe"][i])
            r0 = batch["response_0"][i]
            r1 = batch["response_1"][i]

            if name == "safe":
                if r0_safe:
                    msgs.append(
                        [
                            {"role": "user", "content": p},
                            {"role": "assistant", "content": r0},
                        ]
                    )
                if r1_safe:
                    msgs.append(
                        [
                            {"role": "user", "content": p},
                            {"role": "assistant", "content": r1},
                        ]
                    )
            else:  # name == "unsafe"
                if not r0_safe:
                    msgs.append(
                        [
                            {"role": "user", "content": p},
                            {"role": "assistant", "content": r0},
                        ]
                    )
                if not r1_safe:
                    msgs.append(
                        [
                            {"role": "user", "content": p},
                            {"role": "assistant", "content": r1},
                        ]
                    )

        # When batched=True, return lists for each column
        return {"messages": msgs}

    mapped = dataset.map(
        to_messages,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )

    # Same as opc.py: keep only the train split and perform a 9:1 split
    split = mapped["train"].train_test_split(test_size=0.1, seed=42)
    return split


if __name__ == "__main__":
    from dllm.utils import resolve_with_base_env

    dataset_name_or_path = resolve_with_base_env(
        "PKU-Alignment/PKU-SafeRLHF", "BASE_DATASETS_DIR"
    )
    dataset = load_dataset_pku_rlhf_sft(dataset_name_or_path, "safe")
    breakpoint()
