"""
Local users
------------
python examples/roberta/generate.py --model_name_or_path "YOUR_MODEL_PATH"

Slurm users
------------
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 \
    python examples/roberta/generate.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import tyro
import torch
import transformers

import dllm
from dllm.pipelines import llada


@dataclass
class ScriptArguments:
    # model_name_or_path: str = "models/LLaDA-8B-Base/tiny-shakespeare/checkpoint-final"
    # model_name_or_path: str = "models/roberta-large-scratch/tiny-shakespeare/scheduler/checkpoint-final"
    # model_name_or_path: str = "models/roberta-large/wikitext-103-v1/epochs-50-bs-4096-len-128/checkpoint-11200"
    # model_name_or_path: str = "models/roberta-large/wikitext-103-v1/epochs-50-bs-4096-len-512/checkpoint-final"
    model_name_or_path: str = "models/ModernBERT-large/wikitext-103-v1/epochs-20-bs-512-len-512/checkpoint-3608"
    steps: int = 128
    max_length: int = 128
    block_length: int = 128
    temperature: float = 0.0
    remasking: str = "random"
    cfg_scale: float = 0.0
    seed: int = 42

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


script_args = tyro.cli(ScriptArguments)
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args, model=model)

# --- Example 1: Batch generation ---
print("\n" + "=" * 80)
print("TEST: generate()".center(80))
print("=" * 80)

# import torch
# input_ids_list = [torch.tensor([])]

prompts = [
    # "Here is an educational python function:"
    # "The route forms the main streets of several of the small towns that dot the highway east to west , namely Cookstown , Alliston and Shelburne .",
    "The 2011 – 12 Columbus Blue Jackets season was the team 's",
    # "Boston Celtics is"
]
input_ids_list = [
    tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    for prompt in prompts
]

out = llada.generate(
    model,
    tokenizer,
    input_ids_list,
    steps=script_args.steps,
    max_new_tokens=None,
    max_length=script_args.max_length,
    block_length=script_args.block_length,
    temperature=script_args.temperature,
    remasking=script_args.remasking,
    cfg_scale=script_args.cfg_scale,
)

generations = [g.split(tokenizer.eos_token, 1)[0] for g in tokenizer.batch_decode(out)]
for i, o in enumerate(generations):
    print("\n" + "-" * 80)
    print(f"[Case {i}]")
    print("-" * 80)
    print(o.strip() if o.strip() else "<empty>")

print("\n" + "=" * 80 + "\n")
