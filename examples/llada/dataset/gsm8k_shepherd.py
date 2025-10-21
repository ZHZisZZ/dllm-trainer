from collections import Counter
from datasets import concatenate_datasets, load_dataset, Dataset, load_from_disk
import random
import re, os
from typing import Dict, Tuple, Optional, List
from rapidfuzz import process, fuzz
from difflib import SequenceMatcher
import sys, pathlib
ROOT = pathlib.Path().resolve().parent  # 현재 노트북이 있는 디렉토리의 상위
sys.path.append(str(ROOT))
from gsm_scorer import GSM8KScorer
from math_scorer import MATHScorer

TARGET_SIZE = 120_000
BALANCED_TASKS = ("GSM8K", "MATH")
SEED = 42

def _normalize_task(name: str) -> str:
    return name.strip().lower()

def _count_pm(values):
    """values: list of '+' / '-' (기타 토큰은 무시)"""
    c = Counter(v for v in values if v in {"+", "-"})
    return int(c.get("+", 0)), int(c.get("-", 0))

def _prepare_task_ds(ds: Dataset, task_norm: str, seed: int):
    # task 필터
    tds = ds.filter(lambda ex, task_norm=task_norm: _normalize_task(ex["task"]) == task_norm)
    # 각 예제의 +, - 개수 계산
    tds = tds.map(
        lambda ex: {"plus_cnt": _count_pm(ex["value"])[0], "minus_cnt": _count_pm(ex["value"])[1]},
        desc=f"count +/- for {task_norm}"
    )
    # 셔플(재현성)
    tds = tds.shuffle(seed=seed)
    return tds

def _greedy_balance_pick(tds: Dataset, k: int, seed: int):
    """현재 누적 +/- 차이를 줄이는 방향으로 k개 고르기 (불가능하면 채울 때까지 보충)."""
    # 분기: +가 많은 샘플 / -가 많은 샘플 / 동률
    pos_dominant_idx = [i for i, ex in enumerate(tds) if ex["plus_cnt"] > ex["minus_cnt"]]
    neg_dominant_idx = [i for i, ex in enumerate(tds) if ex["minus_cnt"] > ex["plus_cnt"]]
    tie_idx          = [i for i, ex in enumerate(tds) if ex["minus_cnt"] == ex["plus_cnt"]]

    rng = random.Random(seed)
    rng.shuffle(pos_dominant_idx)
    rng.shuffle(neg_dominant_idx)
    rng.shuffle(tie_idx)

    sel_idx = []
    total_plus = 0
    total_minus = 0

    def take_from(bucket):
        if not bucket: 
            return False
        i = bucket.pop()  # 뒤에서 pop (셔플되어 있음)
        sel_idx.append(i)
        nonlocal total_plus, total_minus
        p, m = tds[i]["plus_cnt"], tds[i]["minus_cnt"]
        total_plus  += p
        total_minus += m
        return True

    while len(sel_idx) < k and (pos_dominant_idx or neg_dominant_idx or tie_idx):
        # 현재 어떤 쪽이 부족한지 보고 선택
        if total_plus <= total_minus:
            # +를 늘리는 쪽을 우선
            if not take_from(pos_dominant_idx):
                if not take_from(tie_idx):
                    take_from(neg_dominant_idx)
        else:
            # -를 늘리는 쪽을 우선
            if not take_from(neg_dominant_idx):
                if not take_from(tie_idx):
                    take_from(pos_dominant_idx)

    # 남았는데도 부족하면(모든 버킷 고갈) 그냥 앞에서 채움
    if len(sel_idx) < k:
        remaining = [i for i in range(len(tds)) if i not in set(sel_idx)]
        rng.shuffle(remaining)
        sel_idx.extend(remaining[: (k - len(sel_idx))])

    sel = tds.select(sel_idx)
    return sel

def select_balanced_subset(dataset, tasks=BALANCED_TASKS, target_size=TARGET_SIZE, seed=SEED, balance_value=True):
    normalized_tasks = [_normalize_task(task) for task in tasks]
    per_task, remainder = divmod(target_size, len(normalized_tasks))
    allocations = {task: per_task for task in normalized_tasks}
    for index in range(remainder):
        allocations[normalized_tasks[index]] += 1

    subsets = []
    actual_allocations = {}

    for raw_task, normalized_task in zip(tasks, normalized_tasks):
        task_ds = dataset.filter(
            lambda example, normalized_task=normalized_task: _normalize_task(example["task"]) == normalized_task
        )
        available = len(task_ds)
        take = min(allocations[normalized_task], available)
        if take < allocations[normalized_task]:
            print(f"Warning: Only {available} examples available for task {raw_task}. Taking all of them.")
        subset = task_ds.shuffle(seed=seed).select(range(take))
        actual_allocations[raw_task] = len(subset)
        subsets.append(subset)

    combined = concatenate_datasets(subsets).shuffle(seed=seed)
    print("Selected counts per task:")
    for task in tasks:
        print(f"  {task}: {actual_allocations.get(task, 0)}")
    print(f"Total selected: {len(combined)} (target was {target_size})")
    return combined

def analyze_value_distribution(dataset):
    counts = Counter()
    for vals in dataset["value"]:
        # value가 list라는 가정; 다른 토큰은 무시
        counts.update(v for v in vals if v in {"+", "-"})
    total = sum(counts.values())
    print("Value token distribution:")
    for label in ["+", "-"]:
        portion = counts[label] / total if total else 0.0
        print(f"  {label}: {counts[label]} ({portion:.2%})")
    others = [k for k in counts.keys() if k not in {"+", "-"}]
    if others:
        print("Other tokens:")
        for k in others:
            portion = counts[k] / total if total else 0.0
            print(f"  {k}: {counts[k]} ({portion:.2%})")
    return counts

# Normalization & parsers
def _normalize_text(s: str) -> str:
    if s is None: return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _normalize_task(name: str) -> str:
    return (name or "").strip().lower()

def _extract_question_from_ms_input(ms_input: str) -> Tuple[str, str]:
    if not isinstance(ms_input, str):
        return "", ""
    s = ms_input
    sol = ms_input

    # 1) 첫 'Step <num>:' 이전까지
    m = re.search(r"\bStep\s*\d+\s*:", s, flags=re.IGNORECASE)
    cut = m.start() if m else None
    # 2) 보조 컷 포인트: 'The answer is:'
    m2 = re.search(r"\bThe\s+answer\s+is\s*:", s, flags=re.IGNORECASE)
    cut2 = m2.start() if m2 else None
    # 가장 이른 컷 포인트 선택
    candidates = [x for x in [cut, cut2] if x is not None]
    if candidates:
        s = s[: min(candidates)]
        sol = sol[min(candidates):]
    # 3) 혹시 라벨이 전혀 없고 줄바꿈만 있는 경우: 첫 문단 사용
    # s = s.strip().splitlines()[0] if "\n" in s and len(s.strip().splitlines()[0]) > 20 else s
    # 4) 트레일러 토큰 정리 (마커/수식 시작 전)
    s = s.split(" ки")[0]  # 러시아어 '키' 마커가 문장 뒤에 붙는 케이스 방지
    s = s.split("<<")[0]   # 계산 마커 이전까지만
    s = _normalize_text(s)
    sol = sol.replace(" ки", "")  # 러시아어 '키' 마커들 모두 제거
    sol = _normalize_text(sol)
    return s, sol

# Source indices (train)
def build_gsm8k_index(gsm_train: Dataset) -> Tuple[Dict[str, str], List[str]]:
    q2data, questions = {}, []
    for i, ex in enumerate(gsm_train):
        q = _normalize_text(ex.get("question", ""))
        a_raw = ex.get("answer", None)
        a = GSM8KScorer.extract_gold(a_raw) if isinstance(a_raw, str) else None
        if q and a:
            # Extract solution (everything before ####)
            solution = a_raw.split("####")[0].strip() if "####" in a_raw else a_raw.strip()

            q2data[q] = {
                "index": i,
                "gold_answer": a,
                "solution": solution
            }
            questions.append(q)
    return q2data, questions
    # q2a, questions = {}, []
    # for ex in gsm_train:
    #     q = _normalize_text(ex.get("question", ""))
    #     a_raw = ex.get("answer", None)
    #     a = GSM8KScorer.extract_gold(a_raw) if isinstance(a_raw, str) else None
    #     if q and a:
    #         q2a[q] = a
    #         questions.append(q)
    # return q2a, questions

def build_math500_index(math500_test: Dataset) -> Tuple[Dict[str, str], List[str]]:
    q2a, questions = {}, []
    for ex in math500_test:
        q = _normalize_text(ex.get("problem", ""))
        a = ex.get("answer", None)  # 이미 최종 정답
        if q and isinstance(a, str) and a.strip():
            q2a[q] = _normalize_text(a)
            questions.append(q)
    return q2a, questions

def build_math_main_index(math_train, math_test, math_scorer):
    """hendrycks/competition_math의 train+test에서 (q -> gold) 인덱스 구축"""
    def _one_split(ds):
        q2a, qs = {}, []
        for ex in ds:
            q = _normalize_text(ex.get("problem", ""))
            sol = ex.get("solution", "")
            try:
                a = math_scorer.extract_gold(sol)
            except Exception:
                a = None
            if q and a is not None:
                a = _normalize_text(str(a))
                q2a[q] = a
                qs.append(q)
        return q2a, qs

    q2a_tr, qs_tr = _one_split(math_train)
    q2a_te, qs_te = _one_split(math_test)
    q2a = {**q2a_tr, **q2a_te}
    qs   = qs_tr + qs_te
    return q2a, qs

# Matching
def _best_fuzzy_rf(query: str, candidates: List[str], cutoff: int = 95) -> Optional[str]:
    res = process.extractOne(
        query,
        candidates,
        scorer=fuzz.ratio,     # difflib 유사도와 가장 근접
        score_cutoff=cutoff    # 0~100
        # processor=None  # 이미 외부에서 _normalize_text 했으니 별도 processor 불필요
    )
    return res[0] if res else None

# Main: attach gold_answer with fail indices
def attach_gold_answers_by_task_rf(ms_ds: Dataset, gsm_index: Tuple[Dict[str, str], List[str]], fuzzy_cutoff: int = 95, num_proc: int = os.cpu_count(),) -> Tuple[Dataset, List[int]]:
    gsm_q2a, gsm_qs = gsm_index
    # mmain_q2a, mmain_qs   = math_main_index
    # m5_q2a, m5_qs         = math500_index

    # mapper는 프로세스 간 공유상태 사용 금지 → 실패 여부는 컬럼으로 반환
    def _mapper(example, idx):
        q, sol = _extract_question_from_ms_input(example.get("input", ""))
        t = _normalize_task(example.get("task", ""))

        gold,solution, src, mtype, index = None, None, "", "", -1  # index 기본값 추가
        gsm8k_q = ""
        llm_answer = ""
        if q:
            if t == "gsm8k":
                if q in gsm_q2a:
                    data = gsm_q2a[q]
                    gsm8k_q = q   
                    gold, solution, src, mtype, index = data["gold_answer"], data["solution"], "gsm8k", "exact", data["index"]
                    llm_answer = sol

                else:
                    hit = _best_fuzzy_rf(q, gsm_qs, cutoff=fuzzy_cutoff)
                    if hit: 
                        data = gsm_q2a[hit]
                        gsm8k_q = hit
                        gold, solution, src, mtype, index = data["gold_answer"], data["solution"], "gsm8k", "fuzzy", data["index"]
                        llm_answer = sol

            elif t != "gsm8k" and t != "math":
                # 비정형 태스크: 세 소스 모두 탐색
                if q in gsm_q2a:
                    data = gsm_q2a[q]
                    gsm8k_q = q   
                    gold, solution, src, mtype, index = data["gold_answer"], data["solution"], "gsm8k", "exact", data["index"]
                    llm_answer = sol
                else:
                    hit = _best_fuzzy_rf(q, gsm_qs, cutoff=fuzzy_cutoff)
                    if hit: 
                        data = gsm_q2a[hit]
                        gsm8k_q = hit
                        gold, solution, src, mtype, index = data["gold_answer"], data["solution"], "gsm8k", "fuzzy", data["index"]
                        llm_answer = sol
        return {"index":index, "question":gsm8k_q, "LLM_answer": llm_answer, "gold_answer": gold, "gold_solution": solution, "match_source": src, "match_type": mtype, "match_ok": gold is not None}


    enriched = ms_ds.map(_mapper, with_indices=True, num_proc=num_proc,
                         desc=f"Adding gold_answer (RapidFuzz, cutoff={fuzzy_cutoff})")
    fail_ids = [i for i, ok in enumerate(enriched["match_ok"]) if not ok]
    enriched = enriched.remove_columns(["match_ok"])
    return enriched, fail_ids

# def main():
#     # extract small set
#     ds = load_dataset("zhuzilin/Math-Shepherd")
#     small_train = select_balanced_subset(ds["train"], balance_value=True)
#     value_counts = analyze_value_distribution(small_train)

#     out_dir = "/home/leena/ccc_eval/rs_prm/data/test/ms_small_train"
#     small_train.save_to_disk(out_dir)
#     ms_small = load_from_disk(out_dir)
#     print(f"[SAVE] Dataset saved and reloaded: {len(ms_small)} samples")

#     # attach gold answers
#     gsm_train = load_dataset("openai/gsm8k", "main")["train"]
#     math500 = load_dataset("HuggingFaceH4/MATH-500")["test"] 
#     math_main  = load_dataset("HuggingFaceTB/MATH", "all")
#     math_train = math_main["train"]
#     math_test  = math_main["test"]

#     gsm_index = build_gsm8k_index(gsm_train)
#     math_main_idx  = build_math_main_index(math_train, math_test, MATHScorer)
#     math500_index = build_math500_index(math500)

#     # ms_with_gold, fail_ids = attach_gold_answers_by_task(ms_small, gsm_index=gsm_index, math500_index=math500_index, fuzzy_threshold=0.95)
#     ms_with_gold, fail_ids = attach_gold_answers_by_task_rf(
#         ms_small,
#         gsm_index=gsm_index,
#         math_main_index=math_main_idx,
#         math500_index=math500_index,
#         fuzzy_cutoff=95,
#         # num_proc=os.cpu_count()  # 병렬
#         num_proc=16
#     )

#     out_dir2 = "/home/leena/ccc_eval/rs_prm/data/test/ms_small_train_gold"
#     ms_with_gold.save_to_disk(out_dir2)

#     n_total = len(ms_with_gold)
#     n_fail  = len(fail_ids)
#     n_exact = sum(1 for t in ms_with_gold["match_type"] if t == "exact")
#     n_fuzzy = sum(1 for t in ms_with_gold["match_type"] if t == "fuzzy")
#     print(f"[REPORT] total={n_total}, exact={n_exact}, fuzzy={n_fuzzy}, fail={n_fail}")
#     if n_fail:
#         print(f"[REPORT] fail indices (count={n_fail}): {fail_ids[:50]}{'...' if n_fail > 50 else ''}")

def main():
    # 0) Math-Shepherd 전체 로드 (이미 저장된 폴더에서)
    ds = load_dataset("zhuzilin/Math-Shepherd")
    ms_all = ds['train']
    print(f"[LOAD] Math-Shepherd loaded: {len(ms_all)} samples")

    # 1) 소스 인덱스 준비 (GSM8K, MATH main mirror, MATH-500)
    gsm_train = load_dataset("openai/gsm8k", "main")["train"]
    gsm_index = build_gsm8k_index(gsm_train)
    # math_main  = load_dataset("HuggingFaceTB/MATH", "all")
    # math_train = math_main["train"]
    # math_test  = math_main["test"]
    # math_main_idx  = build_math_main_index(math_train, math_test, MATHScorer)
    # math500 = load_dataset("HuggingFaceH4/MATH-500")["test"]
    # math500_index = build_math500_index(math500)

    # 2) gold_answer 먼저 부착 (RapidFuzz + 병렬)
    ms_with_gold, fail_ids = attach_gold_answers_by_task_rf(
        ms_all,
        gsm_index=gsm_index,
        # math_main_index=math_main_idx,
        # math500_index=math500_index,
        fuzzy_cutoff=95,
        num_proc=16
    )
    print(f"[GOLD] attached: ok={len(ms_with_gold) - len(fail_ids)}, fail={len(fail_ids)}")

    # 중간 저장
    mid_dir = "/home/minhae/diffusion/dllm/examples/llada/dataset/test_ms_with_gold_all"
    os.makedirs(mid_dir, exist_ok=True)
    ms_with_gold.save_to_disk(mid_dir)
    ms_with_gold = load_from_disk(mid_dir)

    # 3) gold 없는 샘플 drop
    def _has_gold(ex):
        g = ex.get("gold_answer", None)
        return isinstance(g, str) and len(g.strip()) > 0
    ms_gold_only = ms_with_gold.filter(_has_gold, num_proc=16)
    print(f"[FILTER] kept gold-only: {len(ms_gold_only)}")

    # # 4) 이제 ‘남은 데이터셋’에서 균형 추출 (GSM8K:MATH=1:1, +/- 밸런스도 그리디로 맞춤)
    # TARGET = 120_000
    # balanced120k = select_balanced_subset(
    #     ms_gold_only,
    #     tasks=BALANCED_TASKS,     # ("GSM8K","MATH")
    #     target_size=TARGET,
    #     seed=SEED,
    #     balance_value=True
    # )

    # 5) 통계 출력 및 저장
    _ = analyze_value_distribution(ms_gold_only)
    out_dir = "/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_match_index_llm_answer2"
    os.makedirs(out_dir, exist_ok=True)
    ms_gold_only.save_to_disk(out_dir)
    print("[DONE] Final dataset:", len(ms_gold_only))


if __name__ == "__main__":
    main()

