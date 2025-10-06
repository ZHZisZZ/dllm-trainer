import re
import os
import tyro
import torch
from dataclasses import dataclass, asdict
from typing import Optional, Annotated

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# ✅ Import the generate function and GenCfg from your generate.py
from examples.editflow.generate import generate_editflow_minimal, GenCfg

DEBUG = True
# -------------------- CONFIG --------------------

MODEL_PATH = "/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/EditFlow-LLaDA-8B-Instruct/tulu-3-sft-mixture-train10000-test1000-5e-5-sample_x0_mixture/checkpoint-final"
DATASET_PATH_TEMPLATE = "/mnt/lustrenew/mllm_safety-shared/datasets/huggingface/cais/mmlu/{field}"

# Prompt template for MMLU
PROMPT_TEMPLATE = """
You are given a multiple-choice question. Select the single best answer from the choices.

Question:
{question}

Choices:
{choices_str}

Answer with your reasoning and the number of the correct choice.
"""

# -------------------- ARGS STRUCT --------------------

@dataclass
class ScriptArgs:
    model_name_or_path: Annotated[str, "Path or hub id for the model"]
    prompt: Annotated[Optional[str], "Text prompt. Will be replaced dynamically"] = None
    field: str = "high_school_computer_science" 
    time_independent: bool = True
    edit_prompt: bool = False
    tau: float = 0.01
    time_epsilon: float = 1e-3
    mask_length: int = 128
    temperature: float = 0.7
    seed: int = 1234
    verbose: bool = False
    make_gif: bool = False
    gif_path: Optional[str] = None
    frame_ms: int = 120
    output_dir: str = "./results"   # ✅ New argument for saving results


# -------------------- MAIN --------------------

def main():
    # 1️⃣ Parse CLI args
    args = tyro.cli(ScriptArgs)

    # 2️⃣ Load model and tokenizer
    print(f"[load model] {MODEL_PATH}")
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 3️⃣ Prepare generation config
    cfg = GenCfg(
        tau=args.tau,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
        time_independent=args.time_independent,
    )

    # 4️⃣ Load MMLU dataset
    dataset_path = DATASET_PATH_TEMPLATE.format(field=args.field)
    dataset = load_dataset(dataset_path)
    test_set = dataset["test"]

    total = 0
    correct = 0

    # 5️⃣ Loop over questions and generate answers
    for i, sample in enumerate(test_set):
        question = sample["question"]
        choices = sample["choices"]
        gt_idx = sample["answer"] + 1
        choices_str = "\n".join(f"{j+1}. {c}" for j, c in enumerate(choices))

        prompt = PROMPT_TEMPLATE.format(
            question=question,
            choices_str=choices_str
        )
        args.prompt = prompt  # dynamically inject prompt

        final_text, trace = generate_editflow_minimal(model, tokenizer, args, cfg)

        tail = final_text[len(prompt):]
        m = re.search(r"\b([1-4])\b", tail)
        pred_idx = int(m.group(1)) if m else None

        total += 1
        if pred_idx == gt_idx:
            correct += 1

        if DEBUG:
            print("=" * 80)
            print(f"[Q{i}] {question}")
            print(prompt)
            print(f"[Generated answer] {final_text.strip()}\n")
            print(f"[Correct Answer]:{sample['answer']+1} {sample['choices'][sample['answer']]}")
            print(f"[Result] {'✅ Correct' if pred_idx == gt_idx else '❌ Wrong'}")
            
            breakpoint()

    print("\n" + "=" * 80)
    acc = correct / total
    model_name = os.path.basename(os.path.dirname(args.model_name_or_path.rstrip("/")))
    print(f"✅ Final Accuracy on MMLU ({total} questions) | Model: {model_name} | mask_length={args.mask_length} → {acc:.2%}")

    # 6️⃣ Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{model_name}_mask{args.mask_length}_{args.field}.txt")

    with open(save_path, "w") as f:
        f.write("==== Experiment Arguments ====\n")
        for k, v in asdict(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\n==== Final Results ====\n")
        f.write(f"Accuracy: {acc:.2%} ({correct}/{total})\n")

    print(f"📁 Results saved to: {save_path}")


if __name__ == "__main__":
    main()
