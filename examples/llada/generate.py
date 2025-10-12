"""
Local users
------------
python examples/llada/generate.py --model_name_or_path "YOUR_MODEL_PATH"

Slurm users
------------
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 \
    python examples/llada/generate.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import tyro
import transformers

import dllm
from dllm.pipelines import llada


@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        "GSAI-ML/LLaDA-8B-Instruct"  # "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    )
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
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
print("TEST: llada.generate()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

input_ids_list = [
    tokenizer.apply_chat_template(
        m,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].to(model.device)
    for m in messages
]

out = llada.generate(
    model,
    tokenizer,
    input_ids_list,
    steps=script_args.steps,
    max_new_tokens=script_args.max_new_tokens,
    block_length=script_args.block_length,
    temperature=script_args.temperature,
    remasking=script_args.remasking,
)

generations = [g.split(tokenizer.eos_token, 1)[0] for g in tokenizer.batch_decode(out)]
for i, o in enumerate(generations):
    print("\n" + "-" * 80)
    print(f"[Case {i}]")
    print("-" * 80)
    print(o.strip() if o.strip() else "<empty>")

print("\n" + "=" * 80 + "\n")

# --- Example 2: Batch fill-in-the-blanks ---
print("\n" + "=" * 80)
print("TEST: llada.infilling()".center(80))
print("=" * 80)

masked_inputs = [
    [
        {"role": "user", "content": tokenizer.mask_token * 20},
        {
            "role": "assistant",
            "content": "Sorry, I do not have answer to this question.",
        },
    ],
    [
        {"role": "user", "content": "Please write an educational python function."},
        {
            "role": "assistant",
            "content": "def hello_" + tokenizer.mask_token * 20 + " return",
        },
    ],
]

fib_input_ids_list = [
    tokenizer.apply_chat_template(
        m,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )[0].to(model.device)
    for m in masked_inputs
]

out = llada.infilling(
    model,
    tokenizer,
    fib_input_ids_list,
    steps=script_args.steps,
    temperature=script_args.temperature,
    remasking=script_args.remasking,
)

filled = tokenizer.batch_decode(out)

for i, (ids, f) in enumerate(zip(fib_input_ids_list, filled)):
    print("\n" + "-" * 80)
    print(f"[Case {i}]")
    print("-" * 80)
    print("[Masked]:\n" + tokenizer.decode(ids))
    print("\n[Filled]:\n" + (f.strip() if f.strip() else "<empty>"))

print("\n" + "=" * 80 + "\n")
