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
    del_rate_hat: [B, T]
    sub_rate_hat: [B, T]
    ins_rate_hat: [B, T]
    sub_logits  : [B, T, V]
    ins_logits  : [B, T, V]
Insertions are applied to the RIGHT of token i.

Local users
------------
python scripts/examples/llada_generate.py --model_name_or_path "YOUR_MODEL_PATH" --tau 0.01 --mask_length 128 --prompt "write an educational python function"

Slurm users
------------
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 \
    python scripts/examples/editflow_generate.py --model_name_or_path "YOUR_MODEL_PATH" --tau 0.01 --mask_length 128 --prompt "write an educational python function"
"""
# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 python scripts/examples/editflow_generate.py --model_name_or_path models/EditFlow-Dream-7B/opc-sft-stage2/checkpoint-final --tau 0.01 --mask_length 128 --seed 3333 --prompt "write an educational python function"
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Annotated

import tyro
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dllm.pipelines.editflow.schedulers import BaseKappaScheduler, LinearKappaScheduler


# ------------------------------- Small utilities --------------------------------

def _bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """First-order τ-leap Bernoulli with p ≈ rate * τ (clamped)."""
    p = (rate * tau).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)

def _sample_from_logits(logits_row: torch.Tensor, temperature: float) -> int:
    """Sample one token id from a 1D logits row with temperature.
    temperature <= 0 -> greedy (argmax).
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    return int(torch.distributions.Categorical(logits=logits_row / temperature).sample().item())


@dataclass
class GenCfg:
    tau: float = 0.02                  # τ step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    edit_prompt: bool = False          # allow editing inside prompt region?
    temperature: float = 0.7           # token sampling temperature (sub/ins)
    verbose: bool = True               # whether to show intermediate decoding traces


# -------------------------------- τ-leap one step --------------------------------

@torch.no_grad()
def tau_leap_step_minimal(
    x: torch.Tensor,            # [T]
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_len: int,            # number of initial prompt tokens (including BOS)
    t: float,
    sched: BaseKappaScheduler,
    cfg: GenCfg,
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
    device = model.device
    T = x.numel()

    attn = torch.ones(1, T, dtype=torch.long, device=device)
    t = torch.full((1, 1), float(t), device=device)

    out = model(input_ids=x.unsqueeze(0), attention_mask=attn, t=t)

    del_rate_h = out["del_rate_hat"]      # [1, T]
    sub_rate_h = out["sub_rate_hat"]      # [1, T]
    ins_rate_h = out["ins_rate_hat"]      # [1, T]
    sub_logits = out["sub_logits"]        # [1, T, V]
    ins_logits = out["ins_logits"]        # [1, T, V]

    # Scale normalized rates to true rates
    tt = torch.tensor([[t]], device=device)
    w = sched.scaling_factor(tt)          # shape [1] or [1,1] depending on impl
    del_rate = del_rate_h * w
    sub_rate = sub_rate_h * w
    ins_rate = ins_rate_h * w

    # Clamp prompt_len within current T (robustness if user passes odd inputs)
    prompt_len_clamped = int(max(1, min(prompt_len, T)))

    # Disable deleting and substituting bos token
    del_rate[:, 0] = 0.0
    sub_rate[:, 0] = 0.0

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

    # Sample token draws where needed (temperature applied here)
    sub_samples: List[Optional[int]] = [
        _sample_from_logits(sub_logits[0, i], cfg.temperature) if choose_sub[i] else None
        for i in range(T)
    ]
    ins_samples: List[Optional[int]] = [
        _sample_from_logits(ins_logits[0, i], cfg.temperature) if ins_fire[i] else None
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

    # print all events
    if cfg.verbose and (comb_fire.any() or ins_fire.any()):
        def _tok_str(tok_id: int) -> str:
            try:
                s = tokenizer.decode([int(tok_id)])
                return s if s.strip() else f"<{int(tok_id)}>"
            except Exception:
                return f"<{int(tok_id)}>"
        ops = []
        for i in range(T):
            if choose_del[i]:
                ops.append(f"DEL@{i}:{_tok_str(int(x[i]))}")
            elif choose_sub[i]:
                ops.append(f"SUB@{i}:{_tok_str(int(x[i]))}->{_tok_str(sub_samples[i])}")
            if ins_samples[i] is not None:
                ops.append(f"INS@{i}->{i+1}:{_tok_str(ins_samples[i])}")
        print("[time]", f"{t.item():.4f}")
        print("[events]", "; ".join(ops))
        print("[decode]\n", tokenizer.decode(new_ids, skip_special_tokens=True))
        print("\n")

    x_next = torch.tensor(new_ids, dtype=torch.long, device=device)

    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())
    return x_next, any_edit


# -------------------------------- top-level generate -------------------------------

@torch.no_grad()
def generate_editflow_minimal(model, tokenizer, args, cfg: GenCfg) -> str:
    device = model.device
    torch.manual_seed(cfg.seed)

    # If prompt is None, start from BOS alone; otherwise ALWAYS prefix BOS
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer must have a BOS token for this sampler.")

    prompt = args.prompt
    if prompt is None:
        ids = [bos]  # BOS alone
    else:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt, add_special_tokens=False)
        ids = [bos] + enc["input_ids"]  # ALWAYS prefix BOS
    
    prompt_len = len(ids)

    if args.mask_length:
        ids = ids + [tokenizer.mask_token_id] * args.mask_length

    x = torch.tensor(ids, dtype=torch.long, device=device)

    sched = LinearKappaScheduler()
    tau = cfg.tau
    # Prefer exact march to t=1 unless capped by max_steps
    steps = math.ceil(1.0 / max(tau, 1e-9))

    t = 0.0
    for step in range(steps):
        # print(step)
        x, edited = tau_leap_step_minimal(
            x=x, 
            model=model, 
            tokenizer=tokenizer, 
            prompt_len=prompt_len, 
            t=t, 
            sched=sched, 
            cfg=cfg
        )
        t = min(1.0, t + tau)
        if t >= 1.0 - args.time_epsilon:  # TODO: modify this to be more general
            break

    # Decode skipping special tokens (keep BOS from leaking)
    print("[final]")
    return tokenizer.decode(x.tolist(), skip_special_tokens=True)


# ---------------------------------------- CLI -------------------------------------

def main():
    @dataclass
    class ScriptArgs:
        # Required (no default)
        model_name_or_path: Annotated[str, "Path or hub id for the model"]

        prompt: Annotated[Optional[str], "Text prompt. If None, start from BOS alone."] = None
        # Boolean flag: tyro exposes --edit-prompt / --no-edit-prompt automatically for bools
        edit_prompt: Annotated[bool,
            "Allow delete/substitute and insertions in the prompt region (BOS+prompt)."] = False

        # Generation-related args
        tau: Annotated[float, "τ-leap size"] = 0.01
        time_epsilon: Annotated[float,
            "Match this with the `time_epsilon` arg used in your EditFlowTrainer"] = 1e-3
        mask_length: Annotated[
            int,
            "Number of <mask> tokens appended after the prompt.\n"
            "EditFlow will iteratively substitute, insert, or delete masks to form the output."
        ] = 128
        temperature: Annotated[float, "Token sampling temperature; 0 for greedy."] = 0.7

        seed: Annotated[int, "Random seed"] = 1234
        verbose: Annotated[bool,
            "Whether to show intermediate decoding traces"] = True


    args = tyro.cli(ScriptArgs)

    cfg = GenCfg(
        tau=args.tau,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    text = generate_editflow_minimal(model, tokenizer, args, cfg)
    print(text)


if __name__ == "__main__":
    main()
