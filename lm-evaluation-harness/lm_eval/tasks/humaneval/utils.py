import os
import torch
import warnings
import evaluate as hf_evaluate

# --- Per-rank cache setup (no Accelerate dependency) ---
if torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()           # global rank (0, 1, 2, ...)
    world_size = torch.distributed.get_world_size()
else:
    rank = 0
    world_size = 1

# Determine base cache directory
base_cache = os.environ.get("HF_EVALUATE_CACHE") or os.environ.get("SLURM_TMPDIR")

if base_cache is None:
    warnings.warn(
        "[Warning] Neither HF_EVALUATE_CACHE nor SLURM_TMPDIR is set. "
        "Please export one of them to a writable directory for evaluation cache, e.g.:\n"
        "    export HF_EVALUATE_CACHE=/home/tmp/hf_eval_cache"
    )
else:
    # Build cache path
    job_id = os.getenv("SLURM_JOB_ID", "local")
    hf_eval_cache = os.path.join(base_cache, f"hf_evaluate_job_{job_id}_rank_{rank}")

    # Ensure directory exists
    os.makedirs(hf_eval_cache, exist_ok=True)
    os.environ["HF_EVALUATE_CACHE"] = hf_eval_cache

    print(f"[Eval Cache] Using cache directory: {hf_eval_cache}")
    print(f"[Eval Cache] rank={rank}, world_size={world_size}")


# --- Load and test metric ---
try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e

def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
