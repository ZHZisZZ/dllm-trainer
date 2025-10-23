from dataclasses import dataclass

import tyro
from huggingface_hub import snapshot_download


@dataclass
class ScriptArguments:
    dataset_id: str = "Anthropic/hh-rlhf"
    allow_patterns: str = None


script_args = tyro.cli(ScriptArguments)

# Replace with the dataset repo you want, e.g. "wikitext"
dataset_id = script_args.dataset_id

# Replace with your desired local directory
local_dir = f"/mnt/lustrenew/mllm_aligned/shared/datasets/huggingface/{dataset_id}"

# Download the dataset snapshot
snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",  # 👈 tell HF it's a dataset
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # ensures real files, not symlinks
    allow_patterns=script_args.allow_patterns,
)

print(f"Dataset downloaded to: {local_dir}")
