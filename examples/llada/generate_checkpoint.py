"""
Local users
------------
python examples/llada/generate_checkpoint.py --model_name_or_path "YOUR_MODEL_PATH"

Slurm users
------------
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 \
    python examples/llada/generate_checkpoint.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import tyro
import torch
import transformers
import accelerate

import dllm
from dllm.pipelines import llada

data_name = "gsm8k_filter_unique_1_0_1"

@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        # "/home/minhae/diffusion/dllm/models/LLaDA-8B-SFT/gsm8k-sft-111/checkpoint-795807"  # "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        "GSAI-ML/LLaDA-8B-Instruct"
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

@dataclass
class DataArguments(dllm.utils.DataArguments):
    # dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]" 
    dataset_args: str = data_name # Use our local GSM8K dataset

data_args = tyro.cli(DataArguments)
script_args = tyro.cli(ScriptArguments)
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args, model=model)


# ----- Data loading -----
def custom_apply_chat_template(messages):
    # Don't move to device in multiprocessing context
    # Return as lists to avoid tensor serialization issues
    q_llm_input_ids = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    q_input_ids = tokenizer.apply_chat_template(
        messages[:-2],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()
    return {"q_llm_input_ids": q_llm_input_ids, "q_len": len(q_input_ids)}

def sft_map_fn(row) -> dict:
    return custom_apply_chat_template(row["messages"])

with accelerate.PartialState().local_main_process_first():
    dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
    dataset = dataset["test"]
    print("test dataset length: ", len(dataset))
    dataset = dataset.filter(lambda x: x["source"] == "gsm8k_q_llm_cond")
    print("filtered test dataset length: ", len(dataset))
    results = dataset.map(sft_map_fn, num_proc=data_args.num_proc, remove_columns=dataset.column_names)
    # Move to device after multiprocessing is done
    input_ids_list = [torch.tensor(r["q_llm_input_ids"]).to(model.device) for r in results]
    q_len = [r["q_len"] for r in results]

# --- Example 1: Batch generation ---
print("\n" + "=" * 80)
print("TEST: llada.generate()".center(80))
print("=" * 80)

# Process in batches to avoid OOM
batch_size = 2  # Adjust based on your GPU memory
all_generations = []

for batch_start in range(0, len(input_ids_list), batch_size):
    batch_end = min(batch_start + batch_size, len(input_ids_list))
    print(f"Processing batch {batch_start//batch_size + 1}/{(len(input_ids_list) + batch_size - 1)//batch_size} (samples {batch_start}-{batch_end-1})")
    
    batch_input_ids = input_ids_list[batch_start:batch_end]
    batch_q_len = q_len[batch_start:batch_end]
    
    # Debug: print first sample info
    if batch_start == 0:
        print(f"\nDEBUG: First sample")
        print(f"  Input length: {len(batch_input_ids[0])}")
        print(f"  Question length: {batch_q_len[0]}")
        print(f"  Max new tokens: {script_args.max_new_tokens}")
        print(f"  Decoded input:\n{tokenizer.decode(batch_input_ids[0])}")
        print()
    
    out = llada.generate_single_cfg(
        model,
        tokenizer,
        batch_input_ids,
        batch_q_len,
        steps=script_args.steps,
        max_new_tokens=script_args.max_new_tokens,
        block_length=script_args.block_length,
        temperature=script_args.temperature,
        remasking=script_args.remasking,
        cfg_scale=0.8  # Higher scale to rely more on conditional (question+answer)
    )
    
    batch_generations = [g.split(tokenizer.eos_token, 1)[0] for g in tokenizer.batch_decode(out)]
    for i, o in enumerate(batch_generations):
        # Extract only the generated part (after the last assistant header)
        parts = o.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            generated_only = parts[-1].strip()  # Get the last part after assistant header
        else:
            generated_only = o.strip()
        
        print("\n" + "-" * 80)
        print(f"[Case {batch_start + i}]")
        print("-" * 80)
        print(f"DEBUG: Full output length: {len(o)}")
        print(f"DEBUG: Extracted generated length: {len(generated_only)}")
        print(f"DEBUG: Generated before strip length: {len(parts[-1]) if len(parts) > 1 else 0}")
        print(f"DEBUG: Input shape: {batch_input_ids[i].shape}, Output shape: {out[i].shape}")
        
        # Show first 200 chars of raw generated part
        if len(parts) > 1:
            raw_gen = parts[-1]
            print(f"DEBUG: Raw generated (first 200 chars): {repr(raw_gen[:200])}")
        
        print(f"\nGenerated text:\n{generated_only if generated_only else '<empty>'}")
    all_generations.extend(batch_generations)
        
    # Clear cache to free memory
    torch.cuda.empty_cache()
    break

# Print first few results
# for i, o in enumerate(all_generations[:5]):
#     print("\n" + "-" * 80)
#     print(f"[Case {i}]")
#     print("-" * 80)
#     print(o.strip() if o.strip() else "<empty>")

print(f"\nTotal generated: {len(all_generations)} samples")
print("\n" + "=" * 80 + "\n")

