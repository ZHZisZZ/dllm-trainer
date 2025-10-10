"""
Minimal EditFlow τ-leap generator for EditBase-Dream with diffusion-style visualization.

What changed vs. your original:
- tau_leap_step_minimal returns (x_next, any_edit, step_trace) preserving all intermediates.
- generate_editflow_minimal returns (final_text, trace).
- render_consecutive_trace_gif(trace, tokenizer, ...) draws a GIF where each frame shows
  ONLY the current output (like the Gemini diffusion page shows progressive refinement):
    * SUB tokens in the current frame are orange
    * INS tokens in the current frame are blue
    * KEEP tokens are black
    * If any deletions happened in the step, the title shows ⌫N (red)
"""

# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 python examples/editflow/generate.py --model_name_or_path "models/EditFlow-Dream-Instruct-7B/tulu-3-sft-mixture/checkpoint-final"  --tau 0.02 --mask_length 128 --seed 7070  --prompt "write a romantic story" --make_gif

import math
from dataclasses import dataclass
from typing import Annotated

import tyro
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dllm.utils.schedulers import BaseKappaScheduler, LinearKappaScheduler


# ------------------------------- Small utilities --------------------------------


def _bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    """First-order τ-leap Bernoulli with p ≈ rate * τ (clamped)."""
    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)


def _sample_from_logits(logits_row: torch.Tensor, temperature: float) -> int:
    """Sample one token id from a 1D logits row with temperature.
    temperature <= 0 -> greedy (argmax).
    """
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    return int(
        torch.distributions.Categorical(logits=(logits_row / temperature))
        .sample()
        .item()
    )


@dataclass
class GenCfg:
    tau: float = 0.02  # τ step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    edit_prompt: bool = False  # allow editing inside prompt region?
    temperature: float = 0.7  # token sampling temperature (sub/ins)
    verbose: bool = True  # whether to show intermediate decoding traces
    time_independent: bool = True


# -------------------------------- τ-leap one step --------------------------------


@torch.no_grad()
def tau_leap_step_minimal(
    x: torch.Tensor,  # [T]
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_len: int,  # number of initial prompt tokens (including BOS)
    t: float,
    sched: BaseKappaScheduler,
    cfg: GenCfg,
    prev_out: dict | None = None,  # <-- pass prior step's model outputs
    reuse_prev: bool = False,  # <-- if True, reuse prev_out instead of forward()
) -> tuple[torch.Tensor, bool, dict, dict]:
    """
    Single τ-leap step with deletion/substitution conflict resolution
    and right-insert policy.

    Reuse semantics:
      • If cfg.time_independent == True and reuse_prev == True and prev_out is not None,
        we reuse `prev_out` tensors instead of calling model() again.
      • Otherwise we run a fresh forward().

    Viz-only convention:
      • Any local annotated as _Ann[*, "viz-only"] is used only for human-visible
        tracing / debugging (console logs, GIFs) and does not affect generation.
      • Such variables are also prefixed with '_' for quick visual scanning.

    Returns:
      x_next, any_edit, _step_trace, out_for_next (the freshly used model outputs)
    """
    device = x.device
    T = x.numel()

    # Decide whether to reuse the previous forward results
    use_reuse = bool(cfg.time_independent and reuse_prev and (prev_out is not None))
    if use_reuse:
        out = prev_out
    else:
        attn = torch.ones(1, T, dtype=torch.long, device=device)
        t_tensor = torch.full((1, 1), float(t), device=device)
        out = model(input_ids=x.unsqueeze(0), attention_mask=attn, t=t_tensor)

    del_rate_h = out["del_rate_hat"]  # [1, T]
    sub_rate_h = out["sub_rate_hat"]  # [1, T]
    ins_rate_h = out["ins_rate_hat"]  # [1, T]
    sub_logits = out["sub_logits"]  # [1, T, V]
    ins_logits = out["ins_logits"]  # [1, T, V]

    # Scale normalized rates to true rates
    tt = torch.tensor([[t]], device=device)
    w = sched.weight(tt)
    del_rate = del_rate_h * w
    sub_rate = sub_rate_h * w
    ins_rate = ins_rate_h * w

    # Clamp prompt_len within current T (robustness)
    prompt_len_clamped = int(max(1, min(prompt_len, T)))

    if not cfg.edit_prompt:
        # Protect the entire prompt span from del/sub
        del_rate[:, :prompt_len_clamped] = 0.0
        sub_rate[:, :prompt_len_clamped] = 0.0
        # Disallow insertions inside the prompt EXCEPT at the last prompt token
        if prompt_len_clamped >= 2:
            ins_rate[:, : prompt_len_clamped - 1] = 0.0

    # Combined "edit" (delete or substitute) event
    comb_rate = (del_rate + sub_rate).squeeze(0)  # [T]
    comb_fire = _bernoulli_from_rate(comb_rate, cfg.tau).bool()  # [T]

    # If an edit fires at i, choose deletion with prob λ_del/(λ_del+λ_sub)
    p_del = (del_rate.squeeze(0) / (comb_rate + 1e-8)).clamp(0, 1)  # [T]
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire  # [T]
    choose_sub = comb_fire & (~choose_del)  # [T]

    # Insertions (right of token i)
    ins_fire = _bernoulli_from_rate(ins_rate.squeeze(0), cfg.tau).bool()  # [T]

    # Token draws (algorithmic, not viz-only)
    sub_samples: list[int | None] = [
        (
            _sample_from_logits(sub_logits[0, i], cfg.temperature)
            if choose_sub[i]
            else None
        )
        for i in range(T)
    ]
    ins_samples: list[int | None] = [
        _sample_from_logits(ins_logits[0, i], cfg.temperature) if ins_fire[i] else None
        for i in range(T)
    ]

    # Build new sequence left→right (apply insertions to the RIGHT)
    new_ids: list[int] = []

    # --- viz-only per-position labels (for trace/GIF) ---
    _before_ops: Annotated[list[str], "viz-only"] = (
        []
    )  # per 'before' position: DEL/SUB/KEEP
    _after_ops: Annotated[list[str], "viz-only"] = (
        []
    )  # per 'after' token aligned to new_ids: INS/SUB/KEEP

    for i in range(T):
        if choose_del[i]:
            _before_ops.append("DEL")
            # deletion -> no token appended
        elif choose_sub[i]:
            _before_ops.append("SUB")
            new_tok = sub_samples[i]
            new_ids.append(int(new_tok))
            _after_ops.append("SUB")
        else:
            _before_ops.append("KEEP")
            new_ids.append(int(x[i].item()))
            _after_ops.append("KEEP")

        if ins_samples[i] is not None:
            new_ids.append(int(ins_samples[i]))
            _after_ops.append("INS")

    x_next = torch.tensor(new_ids, dtype=torch.long, device=device)
    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())
    # Provide the exact outputs we used this step for the caller to pass forward
    out_for_next = out

    # --- (vis) used only for verbose console trace ---
    if cfg.verbose and (comb_fire.any() or ins_fire.any()):

        def _tok_str(tok_id: int) -> str:  # viz-only helper
            try:
                s = tokenizer.decode([int(tok_id)])
                return s if s.strip() else f"<{int(tok_id)}>"
            except Exception:
                return f"<{int(tok_id)}>"

        _ops_strs: Annotated[list[str], "viz-only"] = []
        for i in range(T):
            if choose_del[i]:
                _ops_strs.append(f"DEL@{i}:{_tok_str(int(x[i]))}")
            elif choose_sub[i]:
                _ops_strs.append(
                    f"SUB@{i}:{_tok_str(int(x[i]))}->{_tok_str(sub_samples[i])}"
                )
            if ins_samples[i] is not None:
                _ops_strs.append(f"INS@{i}->{i+1}:{_tok_str(ins_samples[i])}")
        print("[time]", f"{t:.4f}")
        print("[events]", "; ".join(_ops_strs))
        print("[decode]\n", tokenizer.decode(new_ids, skip_special_tokens=False))
        print()

    # --- (vis) step trace payload (returned; used only for visualization downstream) ---
    _step_trace: Annotated[dict, "viz-only"] = {
        "t": float(t),
        "x_before_ids": [int(i) for i in x.tolist()],
        "x_after_ids": [int(i) for i in new_ids],
        "before_ops": _before_ops,  # viz-only labels
        "after_ops": _after_ops,  # viz-only labels
        # below are algorithmic signals copied for visualization/analysis
        "choose_del": [bool(v) for v in choose_del.tolist()],
        "choose_sub": [bool(v) for v in choose_sub.tolist()],
        "ins_fire": [bool(v) for v in ins_fire.tolist()],
        "sub_samples": [int(s) if s is not None else None for s in sub_samples],
        "ins_samples": [int(s) if s is not None else None for s in ins_samples],
        "prompt_len": prompt_len_clamped,
        "used_reuse": bool(use_reuse),
    }

    return x_next, any_edit, _step_trace, out_for_next


# -------------------------------- top-level generate -------------------------------


@torch.no_grad()
def generate_editflow_minimal(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    args,
    cfg: GenCfg,
) -> tuple[str, dict]:
    """
    Returns:
        final_text, trace

    Notes on annotations:
      • Any local annotated with Annotated[..., "viz-only"] is only used to build
        the decode trace for visualization (e.g., GIF rendering) and has no effect
        on the actual generation. Such variables are also prefixed with '_' to make
        this visually obvious in code.
    """
    torch.manual_seed(cfg.seed)

    # If prompt is None, start from BOS alone; otherwise ALWAYS prefix BOS
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer must have a BOS token for this sampler.")

    prompt = args.prompt
    if prompt is None:
        ids = [bos]  # BOS alone
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # ids = tokenizer.encode(prompt, add_special_tokens=False)
        # ids = [bos] + enc["input_ids"]  # ALWAYS prefix BOS

    prompt_len = len(ids)

    if args.mask_length:
        if getattr(tokenizer, "mask_token_id", None) is None:
            raise ValueError(
                "Tokenizer must define mask_token_id when --mask_length > 0."
            )
        ids = ids + [tokenizer.mask_token_id] * args.mask_length

    x = torch.tensor(ids, dtype=torch.long, device=model.device)

    sched = LinearKappaScheduler()
    tau = cfg.tau
    steps = math.ceil(1.0 / max(tau, 1e-9))

    _trace: Annotated[dict, "viz-only: full decode trace for GIF/inspection"] = {
        "steps": [],
        "init": {
            "t": 0.0,
            "x_ids": [int(i) for i in x.tolist()],
            "prompt_len": int(prompt_len),
        },
        "end_t": 0.0,
    }

    # Local-only reuse: if previous iteration had no edits, reuse its forward.
    prev_out: dict | None = None
    prev_had_edits = True  # first iteration must run a forward

    t = 0.0
    for _ in range(steps):
        # We can reuse prev_out only if the model is declared time-independent
        # and the previous step had NO edits (sequence unchanged).
        reuse_prev = (
            cfg.time_independent and not prev_had_edits and (prev_out is not None)
        )

        x, edited, _step_trace, prev_out = tau_leap_step_minimal(
            x=x,
            model=model,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            t=t,
            sched=sched,
            cfg=cfg,
            prev_out=prev_out,
            reuse_prev=reuse_prev,
        )

        _step_trace: Annotated[dict, "viz-only: per-step intermediates for trace"]
        _trace["steps"].append(_step_trace)

        prev_had_edits = edited

        t = min(1.0, t + tau)
        if t >= 1.0 - args.time_epsilon:
            break

    _trace["end_t"] = float(t)

    final_text = tokenizer.decode(x.tolist(), skip_special_tokens=False)
    print("[final]")
    return final_text, _trace


# ------------------------------ Visualization (NEW) ------------------------------
# Diffusion-style consecutive output: only show the CURRENT output per frame.
# ------------------ Visualization (sanitized, masks stripped) ------------------
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
import re
import unicodedata


def render_consecutive_trace_gif(
    trace: dict,
    tokenizer,
    out_path: str = "decode_trace.gif",
    font_size: int = 25,
    line_spacing: int = 12,
    frame_ms: int = 250,
    final_ms: int = 5000,          # final clean frame duration (ms)
    max_width: int = 1400,
    max_height: int = 3000,
    margin: int = 32,
    title_color=(80, 80, 80),
    text_color=(0, 0, 0),          # base black
    mask_color=(150, 150, 150),
    sub_nonmask_color=(200, 0, 0), # persistent red
    ins_color=(0, 0, 200),         # persistent blue
    del_strike_color=(120, 120, 120),
    events_color=(30, 30, 30),
    box_color=(120, 120, 120),
    bg_color=(255, 255, 255),
):
    """
    Persistent coloring keyed by token *instance* (not token id):
      - Inserted tokens -> BLUE across frames (until deleted/substituted again).
      - Substitution nonmask→nonmask -> RED across frames (until deleted/substituted again).
      - Substitution mask→nonmask -> stays BLACK (no extra color).
    Adds a final clean frame (5s) with no events box.
    """
    from PIL import Image, ImageDraw, ImageFont
    import unicodedata

    # ---------- font ----------
    try:
        font = ImageFont.truetype("assets/JetBrainsMono-VariableFont_wght.ttf", font_size)
    except Exception:
        print(f"fail to load target font")
        font = ImageFont.load_default()

    # ---------- helpers ----------
    def _sanitize_token(s: str) -> str:
        vis_mask_token = "<|m|>"
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("Ċ", "\n").replace("▁", " ").replace("Ġ", " ")
        s = s.replace("\t", "    ")
        s = s.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
        if "mdm_mask" in s.lower():
            # Replace any variant like <|mdm_mask|>, ▁<mdm_mask>, Ġ<mdm_mask>
            s = re.sub(r"<[\|]?\s*mdm_mask\s*[\|]?>", vis_mask_token, s, flags=re.IGNORECASE)
            s = s.replace("mdm_mask", vis_mask_token)
        return s

    def _tok_str(tok_id: int) -> str:
        try:
            s = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
            s = s if s.strip() else f"<{int(tok_id)}>"
        except Exception:
            s = f"<{int(tok_id)}>"
        return s.replace("\n", "\\n")
    
    TOKEN_RE = re.compile(r"\s+|\S+")
    def _wrap_text(draw: ImageDraw.ImageDraw, text: str, width_px: int) -> List[str]:
        if text == "":
            return [""]
        lines: List[str] = []
        for para in text.split("\n"):
            tokens = TOKEN_RE.findall(para)
            cur = ""
            for tok in tokens:
                candidate = cur + tok
                if draw.textlength(candidate, font=font) <= width_px:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                        cur = tok
                        while draw.textlength(cur, font=font) > width_px and len(cur) > 0:
                            lo, hi, fit = 1, len(cur), 1
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                if draw.textlength(cur[:mid], font=font) <= width_px:
                                    fit, lo = mid, mid + 1
                                else:
                                    hi = mid - 1
                            lines.append(cur[:fit])
                            cur = cur[fit:]
                    else:
                        t = tok
                        while draw.textlength(t, font=font) > width_px and len(t) > 0:
                            lo, hi, fit = 1, len(t), 1
                            while lo <= hi:
                                mid = (lo + hi) // 2
                                if draw.textlength(t[:mid], font=font) <= width_px:
                                    fit, lo = mid, mid + 1
                                else:
                                    hi = mid - 1
                            lines.append(t[:fit])
                            t = t[fit:]
                        cur = t
            lines.append(cur)
        return lines or [""]

    tmp_img = Image.new("RGB", (10, 10), bg_color)
    tmp_draw = ImageDraw.Draw(tmp_img)
    text_width_budget = max_width - 2 * margin

    # mask detection
    MASK_IDS = set()
    if getattr(tokenizer, "mask_token_id", None) is not None:
        MASK_IDS.add(int(tokenizer.mask_token_id))
    MASK_STRINGS = set()
    mt = getattr(tokenizer, "mask_token", None)
    if mt is not None:
        MASK_STRINGS.add(str(mt))
    MASK_STRINGS.add("<mdm_mask>")

    def _is_mask_token(tok_id: int, tok_str_exact: str) -> bool:
        return (int(tok_id) in MASK_IDS) or (tok_str_exact in MASK_STRINGS)

    def _wrap_tokens_with_index(tokens, deleted_flags):
        lines, cur, cur_w = [], [], 0
        for i, tok in enumerate(tokens):
            t = _sanitize_token(tok)
            parts = t.split("\n")
            for j, seg in enumerate(parts):
                seg_rest = seg
                while seg_rest:
                    w = tmp_draw.textlength(seg_rest, font=font)
                    if cur_w + w <= text_width_budget or not cur:
                        cur.append((seg_rest, i, deleted_flags[i]))
                        cur_w += w
                        seg_rest = ""
                    else:
                        lines.append(cur); cur, cur_w = [], 0
                if j != len(parts) - 1:
                    lines.append(cur); cur, cur_w = [], 0
        if cur:
            lines.append(cur)
        return lines

    def _draw_dashed_rectangle(draw, xy, dash=8, gap=6, width=2, outline=(120, 120, 120)):
        x0, y0, x1, y1 = xy
        x = x0
        while x < x1:
            x2 = min(x + dash, x1)
            draw.line([(x, y0), (x2, y0)], fill=outline, width=width)
            draw.line([(x, y1), (x2, y1)], fill=outline, width=width)
            x += dash + gap
        y = y0
        while y < y1:
            y2 = min(y + dash, y1)
            draw.line([(x0, y), (x0, y2)], fill=outline, width=width)
            draw.line([(x1, y), (x1, y2)], fill=outline, width=width)
            y += dash + gap

    def _ops_lines_for_step(st: dict):
        if st is None:
            return ["(no events)"]
        lines = []
        x_before = st["x_before_ids"]
        choose_del = st["choose_del"]
        choose_sub = st["choose_sub"]
        sub_samples = st["sub_samples"]
        ins_samples = st["ins_samples"]
        T = len(x_before)
        for i in range(T):
            if choose_del[i]:
                lines.append(f"DEL@{i}:{_tok_str(int(x_before[i]))}")
            elif choose_sub[i]:
                lines.append(
                    f"SUB@{i}:{_tok_str(int(x_before[i]))}->{_tok_str(int(sub_samples[i]))}"
                )
            if ins_samples[i] is not None:
                lines.append(f"INS@{i}->{i+1}:{_tok_str(int(ins_samples[i]))}")
        if not lines:
            lines.append("(no events)")
        return lines

    # ---- Instance-id machinery ----
    next_instance_id = 0
    def _new_inst():
        nonlocal next_instance_id
        val = next_instance_id
        next_instance_id += 1
        return val

    # Current sequence at the *start* (ids + instance_ids)
    curr_ids = list(trace["init"]["x_ids"])
    curr_inst = [ _new_inst() for _ in curr_ids ]

    # Persistent color by instance_id: {"blue", "red"}
    color_by_inst = {}

    # ---------- PASS 1: measure required heights per frame ----------
    measurement_payload = []

    for step_idx, st in enumerate([None] + trace["steps"]):
        # build augmented view
        if st is None:
            aug_ids = list(curr_ids)
            deleted_flags = [False] * len(aug_ids)
        else:
            x_before = st["x_before_ids"]
            choose_del = st["choose_del"]
            after_ids = st["x_after_ids"]
            deleted_positions = [i for i, d in enumerate(choose_del) if d]

            aug_ids = list(after_ids)
            deleted_flags = [False] * len(after_ids)
            for i in sorted(deleted_positions, reverse=True):
                aug_ids.insert(i, x_before[i])
                deleted_flags.insert(i, True)

        tokens = tokenizer.convert_ids_to_tokens(aug_ids)
        wrapped_lines = _wrap_tokens_with_index(tokens, deleted_flags)

        # estimate ops lines for this step
        if st:
            ops_text = "  • " + "  • ".join(_ops_lines_for_step(st))
        else:
            ops_text = "(no events)"
        ops_lines = _wrap_text(tmp_draw, ops_text, text_width_budget)

        # compute height needed
        body_h = len(wrapped_lines) * (font_size + line_spacing)
        ops_h = len(ops_lines) * (font_size + line_spacing) + font_size + 20
        required_h = margin + (font_size + line_spacing) + body_h + 20 + ops_h


        # compute height needed
        body_h = len(wrapped_lines) * (font_size + line_spacing)
        ops_h = len(ops_lines) * (font_size + line_spacing) + font_size + 20
        required_h = margin + (font_size + line_spacing) + body_h + 20 + ops_h

        measurement_payload.append(
            {
                "step_idx": step_idx,
                "st": st,
                "aug_ids": aug_ids,
                "tokens": tokens,
                "deleted_flags": deleted_flags,
                "wrapped_lines": wrapped_lines,
                "ops_lines": ops_lines,
                "required_h": required_h,
            }
        )

    # Measure clean final frame (no events)
    final_text_ids = trace["steps"][-1]["x_after_ids"] if trace["steps"] else trace["init"]["x_ids"]
    final_tokens = tokenizer.convert_ids_to_tokens(final_text_ids)
    wrapped_clean = _wrap_tokens_with_index(final_tokens, [False] * len(final_tokens))
    clean_body_h = len(wrapped_clean) * (font_size + line_spacing)
    clean_required_h = margin + (font_size + line_spacing) + clean_body_h

    # Pick a single uniform canvas height
    max_required_h = max([p["required_h"] for p in measurement_payload] + [clean_required_h]) + 20
    H = min(max_required_h, max_height)
    W = max_width

    # For each frame we need an augmented view (with deleted placeholders) to draw
    frames = []

    # Iterate steps; for step_idx==0 we still draw "initial state"
    steps_with_initial = [None] + trace["steps"]

    for step_idx, st in enumerate(steps_with_initial):
        if st is None:
            # initial frame: augmented is just current tokens
            aug_ids = list(curr_ids)
            aug_inst = list(curr_inst)
            aug_deleted = [False] * len(aug_ids)
            ops_lines = ["(no events)"]
            title = "initial state"
        else:
            title = f"t = {st['t']:.3f}"
            x_before = list(st["x_before_ids"])
            choose_del = list(st["choose_del"])
            choose_sub = list(st["choose_sub"])
            sub_samples = list(st["sub_samples"])
            ins_samples = list(st["ins_samples"])
            assert len(x_before) == len(curr_ids) == len(curr_inst), "trace 'x_before' must match current sequence."

            # Build augmented (drawn) and next (state-after) in one pass
            aug_ids, aug_inst, aug_deleted = [], [], []
            next_ids, next_inst = [], []

            for i in range(len(x_before)):
                before_id = int(curr_ids[i])
                before_inst = curr_inst[i]

                if choose_del[i]:
                    # show deleted placeholder (strike-through)
                    aug_ids.append(before_id); aug_inst.append(None); aug_deleted.append(True)
                    # remove from next; also clear any persistent color
                    color_by_inst.pop(before_inst, None)
                else:
                    if choose_sub[i]:
                        after_id = int(sub_samples[i])
                        # in augmented we show the *after* token at same instance
                        aug_ids.append(after_id); aug_inst.append(before_inst); aug_deleted.append(False)
                        next_ids.append(after_id); next_inst.append(before_inst)

                        # update persistence by source type
                        if int(before_id) in MASK_IDS:
                            # mask → nonmask: no extra color (ensure cleared)
                            color_by_inst.pop(before_inst, None)
                        else:
                            # nonmask → nonmask: mark RED
                            color_by_inst[before_inst] = "red"
                    else:
                        # keep
                        aug_ids.append(before_id); aug_inst.append(before_inst); aug_deleted.append(False)
                        next_ids.append(before_id); next_inst.append(before_inst)

                # insertion AFTER position i
                if ins_samples[i] is not None:
                    ins_id = int(ins_samples[i])
                    ins_inst = _new_inst()
                    aug_ids.append(ins_id); aug_inst.append(ins_inst); aug_deleted.append(False)
                    next_ids.append(ins_id); next_inst.append(ins_inst)
                    # mark persistent BLUE for this *instance only*
                    color_by_inst[ins_inst] = "blue"

            # commit next state
            curr_ids, curr_inst = next_ids, next_inst
            ops_text = "  • " + "  • ".join(_ops_lines_for_step(st))
            ops_lines = _wrap_text(tmp_draw, ops_text, text_width_budget)


        # ----- render this frame -----
        tokens = tokenizer.convert_ids_to_tokens(aug_ids)
        wrapped_lines = _wrap_tokens_with_index(tokens, aug_deleted)

        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

        y = margin
        draw.text((margin, y), title, fill=title_color, font=font)
        y += font_size + line_spacing

        for line in wrapped_lines:
            x = margin
            for seg_text, tok_idx, is_deleted in line:
                tok_id = int(aug_ids[tok_idx])
                tok_str_exact = tokens[tok_idx]
                inst = aug_inst[tok_idx]

                if is_deleted:
                    # strike deleted — grey masks slightly different if desired
                    strike_color = mask_color if _is_mask_token(tok_id, tok_str_exact) else del_strike_color
                    strike = "".join(ch + "\u0336" for ch in seg_text)
                    draw.text((x, y), strike, fill=strike_color, font=font)
                    x += tmp_draw.textlength(strike, font=font)
                else:
                    # choose color by *instance*
                    color = text_color
                    if inst is not None and inst in color_by_inst:
                        color = ins_color if color_by_inst[inst] == "blue" else sub_nonmask_color
                    elif _is_mask_token(tok_id, tok_str_exact):
                        color = mask_color
                    draw.text((x, y), seg_text, fill=color, font=font)
                    x += tmp_draw.textlength(seg_text, font=font)
            y += font_size + line_spacing

        # draw events box for all but the extra final-clean frame we'll add later
        # if step_idx != len(steps_with_initial) - 1:
        #     y += 20
        #     x0, y0 = margin, y
        #     x1 = max_width - margin
        #     box_h = len(ops_lines) * (font_size + line_spacing) + font_size + 20
        #     y1 = y0 + box_h
        #     _draw_dashed_rectangle(draw, (x0, y0, x1, y1), outline=box_color)
        #     draw.text((x0 + 10, y0 + 10), "events", fill=events_color, font=font)
        #     yy = y0 + font_size + 20
        #     for l in ops_lines:
        #         draw.text((x0 + 10, yy), l, fill=events_color, font=font)
        #         yy += font_size + line_spacing
        # y += 10
        frames.append(img)

    # ----- extra final clean frame (no events box), 5s -----
    final_ids = list(curr_ids)
    final_inst = list(curr_inst)
    final_tokens = tokenizer.convert_ids_to_tokens(final_ids)

    # wrap without deleted flags
    def _wrap_clean(tokens):
        lines, cur, cur_w = [], [], 0
        for i, tok in enumerate(tokens):
            t = _sanitize_token(tok)
            parts = t.split("\n")
            for j, seg in enumerate(parts):
                seg_rest = seg
                while seg_rest:
                    w = tmp_draw.textlength(seg_rest, font=font)
                    if cur_w + w <= text_width_budget or not cur:
                        cur.append((seg_rest, i)); cur_w += w; seg_rest = ""
                    else:
                        lines.append(cur); cur, cur_w = [], 0
                if j != len(parts) - 1:
                    lines.append(cur); cur, cur_w = [], 0
        if cur:
            lines.append(cur)
        return lines

    wrapped_clean = _wrap_clean(final_tokens)

    clean_img = Image.new("RGB", (W, H), bg_color)
    draw = ImageDraw.Draw(clean_img)
    draw.text((margin, margin), "final text", fill=title_color, font=font)
    y = margin + font_size + line_spacing
    for line in wrapped_clean:
        x = margin
        for seg_text, tok_idx in line:
            tok_id = int(final_ids[tok_idx])
            tok_str_exact = final_tokens[tok_idx]
            inst = final_inst[tok_idx]
            color = text_color
            if inst in color_by_inst:
                color = ins_color if color_by_inst[inst] == "blue" else sub_nonmask_color
            elif _is_mask_token(tok_id, tok_str_exact):
                color = mask_color
            draw.text((x, y), seg_text, fill=color, font=font)
            x += tmp_draw.textlength(seg_text, font=font)
        y += font_size + line_spacing
    frames.append(clean_img)

    # save GIF
    durations = [frame_ms] * (len(frames) - 1) + [final_ms]
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return out_path


# ---------------------------------------- CLI -------------------------------------


def main():
    @dataclass
    class ScriptArgs:
        # Required (no default)
        model_name_or_path: Annotated[str, "Path or hub id for the model"]
        time_independent: Annotated[
            bool, "Whether model is conditioned on time step"
        ] = True

        prompt: Annotated[str | None, "Text prompt. If None, start from BOS alone."] = (
            None
        )
        # Boolean flag: tyro exposes --edit-prompt / --no-edit-prompt automatically for bools
        edit_prompt: Annotated[
            bool,
            "Allow delete/substitute and insertions in the prompt region (BOS+prompt).",
        ] = False

        # Generation-related args
        tau: Annotated[float, "τ-leap size"] = 0.01
        time_epsilon: Annotated[
            float, "Match this with the `time_epsilon` arg used in your EditFlowTrainer"
        ] = 1e-3
        mask_length: Annotated[
            int,
            "Number of <mask> tokens appended after the prompt.\n"
            "EditFlow will iteratively substitute, insert, or delete masks to form the output.",
        ] = 128
        temperature: Annotated[float, "Token sampling temperature; 0 for greedy."] = 0.7

        seed: Annotated[int, "Random seed"] = 1234
        verbose: Annotated[bool, "Whether to show intermediate decoding traces"] = True

        # Visualization
        make_gif: Annotated[bool, "Render a decoding trace GIF after generation."] = (
            False
        )
        gif_path: Annotated[
            str | None, "Output GIF path (default: decode_trace.gif)"
        ] = None
        frame_ms: Annotated[int, "Per-frame duration in ms"] = 120

    args = tyro.cli(ScriptArgs)

    cfg = GenCfg(
        tau=args.tau,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
        time_independent=args.time_independent,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    final_text, trace = generate_editflow_minimal(model, tokenizer, args, cfg)
    print(final_text)

    if args.make_gif:
        from examples.editflow.viz import render_consecutive_trace_gif

        out = args.gif_path or "decode_trace.gif"
        path = render_consecutive_trace_gif(
            trace,
            tokenizer,
            out_path=out,
            frame_ms=args.frame_ms,
        )
        print(f"[gif saved] {path}")


if __name__ == "__main__":
    main()
