import os
import torch
import evaluate as hf_evaluate

# --- Per-rank cache setup (no Accelerate dependency) ---
if torch.distributed.is_initialized():
    rank = torch.distributed.get_rank()           # global rank (0, 1, 2, ...)
    world_size = torch.distributed.get_world_size()
else:
    rank = 0
    world_size = 1

cache_root = "/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/.cache"
hf_eval_cache = f"{cache_root}/hf_evaluate_job_{os.getenv('SLURM_JOB_ID', 'local')}_rank_{rank}"
os.environ["HF_EVALUATE_CACHE"] = hf_eval_cache
os.makedirs(hf_eval_cache, exist_ok=True)

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
