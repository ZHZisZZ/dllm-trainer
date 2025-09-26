#!/usr/bin/env python3
"""
Minimal EditFlow τ-leap generator for EditBase-Dream.

Changes requested:
- ALWAYS prefix BOS to the prompt. If --prompt is None, start from BOS alone.
- Add a flag --edit-prompt: when False (default), the initial prompt region (BOS + encoded
  prompt tokens) is *protected*:
    * No delete/substitute on prompt tokens.
    * No insertions inside the prompt, EXCEPT at the last prompt token position
      (right of it) to grow the response.
  When True, prompt tokens can be deleted/substituted and we also allow insertions
  at any prompt position (still inserting to the right). BOS itself remains protected
  from deletion/substitution in all cases.
- Keep everything else minimal and consistent with training:
    * Rates are normalized and scaled by w(t) = κ̇(t)/(1-κ(t)) with κ(t)=t.

Model forward contract (strict):
    out = model(input_ids=x, attention_mask=mask, t=t)
and out must contain:
    del_rate_norm: [B, T]
    sub_rate_norm: [B, T]
    ins_rate_norm: [B, T]
    sub_logits   : [B, T, V]
    ins_logits   : [B, T, V]
Insertions are applied to the RIGHT of token i.
"""
# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 python scripts/examples/editflow_generate.py --model-path models/EditFlow-Dream-7B/checkpoint-final --max-len 10

from dataclasses import dataclass
from typing import Optional, List, Tuple
import argparse
import torch
from transformers import AutoModel, AutoTokenizer

from dllm.pipelines.editflow.schedulers import BaseKappaScheduler, CubicKappaScheduler


# ------------------------------- Small utilities --------------------------------

def _bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """First-order τ-leap Bernoulli with p ≈ rate * τ (clamped)."""
    p = (rate * tau).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)

def _sample_from_logits(logits_row: torch.Tensor) -> int:
    """Sample one token id from a 1D logits row."""
    return int(torch.distributions.Categorical(logits=logits_row).sample().item())


@dataclass
class GenCfg:
    tau: float = 0.02                  # τ step
    max_steps: int = 100               # safety cap (ceil(1/τ) also used)
    max_len: int = 2048                # truncate if it grows too long
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    edit_prompt: bool = False          # allow editing inside prompt region?


# -------------------------------- τ-leap one step --------------------------------

@torch.no_grad()
def tau_leap_step_minimal(
    x: torch.Tensor,            # [T]
    model,
    prompt_len: int,            # number of initial prompt tokens (including BOS)
    t_scalar: float,
    sched: BaseKappaScheduler,
    cfg: GenCfg
) -> Tuple[torch.Tensor, bool]:
    """
    Single τ-leap step with deletion/substitution conflict resolution
    and right-insert policy.

    - BOS (index 0) is always protected from delete/substitute.
    - If cfg.edit_prompt is False, protect the prompt span [0, prompt_len):
        * del/sub disabled for i in [0, prompt_len)
        * insertion disabled for i in [0, prompt_len-1), but allowed at i=prompt_len-1
          (the last prompt token) and anywhere i >= prompt_len.
    """
    device = cfg.device
    T = x.numel()

    attn = torch.ones(1, T, dtype=torch.long, device=device)
    t = torch.full((1, 1), float(t_scalar), device=device)

    out = model(input_ids=x.unsqueeze(0), attention_mask=attn, t=t)

    # Strict naming per user note
    del_rate_n = out["del_rate_hat"]      # [1, T]
    sub_rate_n = out["sub_rate_hat"]      # [1, T]
    ins_rate_n = out["ins_rate_hat"]      # [1, T]
    sub_logits = out["sub_logits"]        # [1, T, V]
    ins_logits = out["ins_logits"]        # [1, T, V]

    # Scale normalized rates to true rates
    tt = torch.tensor([[t_scalar]], device=device)
    # w = (sched.kappa_dot(tt) / (1.0 - sched.kappa(tt))).squeeze(1)  # [1]
    w = sched.scaling_factor(tt)
    del_rate = del_rate_n * w
    sub_rate = sub_rate_n * w
    ins_rate = ins_rate_n * w

    # Clamp prompt_len within current T (robustness if user passes odd inputs)
    prompt_len_clamped = int(max(1, min(prompt_len, T)))

    # # BOS guard: never delete/substitute BOS
    # del_rate[:, 0] = 0.0
    # sub_rate[:, 0] = 0.0

    if not cfg.edit_prompt:
        # Protect the entire prompt span from del/sub
        del_rate[:, :prompt_len_clamped] = 0.0
        sub_rate[:, :prompt_len_clamped] = 0.0

        # Disallow insertions inside the prompt EXCEPT at the last prompt token
        if prompt_len_clamped >= 2:
            ins_rate[:, :prompt_len_clamped-1] = 0.0
        # At index prompt_len_clamped-1 (last prompt token), keep model's ins_rate
        # Positions >= prompt_len_clamped (response area) remain as predicted

    # Combined "edit" (delete or substitute) event
    comb_rate = (del_rate + sub_rate).squeeze(0)                     # [T]
    comb_fire = _bernoulli_from_rate(comb_rate, cfg.tau).bool()      # [T]

    # If an edit fires at i, choose deletion with prob λ_del/(λ_del+λ_sub)
    p_del = (del_rate.squeeze(0) / (comb_rate + 1e-8)).clamp(0, 1)   # [T]
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire        # [T]
    choose_sub = comb_fire & (~choose_del)                           # [T]

    # Insertions (right of token i)
    ins_fire = _bernoulli_from_rate(ins_rate.squeeze(0), cfg.tau).bool()  # [T]

    if torch.any(ins_fire):
        print("ins")

    # Sample token draws where needed
    sub_samples: List[Optional[int]] = [
        _sample_from_logits(sub_logits[0, i]) if choose_sub[i] else None
        for i in range(T)
    ]
    ins_samples: List[Optional[int]] = [
        _sample_from_logits(ins_logits[0, i]) if ins_fire[i] else None
        for i in range(T)
    ]

    # Build new sequence left→right (apply insertions to the RIGHT)
    new_ids: List[int] = []
    for i in range(T):
        if choose_del[i]:
            # drop x[i]
            pass
        elif choose_sub[i]:
            new_ids.append(sub_samples[i])
        else:
            new_ids.append(int(x[i].item()))
        if ins_samples[i] is not None:
            new_ids.append(ins_samples[i])

    x_next = torch.tensor(new_ids, dtype=torch.long, device=device)
    if x_next.numel() > cfg.max_len:
        x_next = x_next[:cfg.max_len]

    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())
    return x_next, any_edit


# -------------------------------- top-level generate -------------------------------

@torch.no_grad()
def generate_editflow_minimal(model, tokenizer, prompt: Optional[str], cfg: GenCfg) -> str:
    device = cfg.device
    torch.manual_seed(cfg.seed)

    # If prompt is None, start from BOS alone; otherwise ALWAYS prefix BOS
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer must have a BOS token for this sampler.")

    if prompt is None:
        ids = [bos]  # BOS alone
    else:
        enc = tokenizer(prompt, add_special_tokens=False)
        ids = [bos] + enc["input_ids"]  # ALWAYS prefix BOS

    x = torch.tensor(ids, dtype=torch.long, device=device)

    # prompt_len counts BOS + encoded prompt tokens, and remains fixed when edit_prompt is False
    prompt_len = len(ids)

    sched = CubicKappaScheduler()
    tau = cfg.tau
    # Prefer exact march to t=1 unless capped by max_steps
    import math
    target_steps = math.ceil(1.0 / max(tau, 1e-9))
    steps = min(cfg.max_steps, target_steps)

    t = 0.0
    noedit = 0
    for _ in range(steps):
        print(_)
        x, edited = tau_leap_step_minimal(x, model, prompt_len, t, sched, cfg)
        t = min(1.0, t + tau)
        if edited:
            noedit = 0
        else:
            noedit += 1
        if t >= 1.0 - 1e-3: # TODO: modify this to be more general
            break

    # Decode skipping special tokens (keep BOS from leaking)
    return tokenizer.decode(x.tolist(), skip_special_tokens=True)


# ---------------------------------------- CLI -------------------------------------

def boolean_flag(parser, name, default=False, help=None):
    """Add --name / --no-name boolean flags."""
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f'--{name}', dest=dest, action='store_true', help=help)
    group.add_argument(f'--no-{name}', dest=dest, action='store_false')
    parser.set_defaults(**{dest: default})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None, help="Text prompt. If None, start from BOS alone.")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    boolean_flag(ap, "edit-prompt", default=False,
                 help="Allow delete/substitute and insertions in the prompt region (BOS+prompt).")
    args = ap.parse_args()

    cfg = GenCfg(
        tau=args.tau,
        max_steps=args.max_steps,
        max_len=args.max_len,
        device=args.device,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
    )

    tok_name = args.tokenizer or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_path).to(cfg.device).eval()

    text = generate_editflow_minimal(model, tokenizer, args.prompt, cfg)
    print(text)


if __name__ == "__main__":
    main()
