from dataclasses import dataclass

import tyro
from huggingface_hub import snapshot_download


@dataclass
class ScriptArguments:
    model_id: str = "GSAI-ML/LLaDA-8B-Instruct"


script_args = tyro.cli(ScriptArguments)

# Replace with the model repo you want, e.g. "bert-base-uncased"
model_id = script_args.model_id

# Replace with your desired local directory
local_dir = f"/mnt/lustrenew/mllm_aligned/shared/models/huggingface/{model_id}"

# Download the model snapshot
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # ensures real files, not symlinks
)

print(f"Model downloaded to: {local_dir}")
