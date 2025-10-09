#!/usr/bin/env python3
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
from typing import Optional, List, Tuple, Annotated

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
    return int(torch.distributions.Categorical(logits=(logits_row / temperature)).sample().item())


@dataclass
class GenCfg:
    tau: float = 0.02                  # τ step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234
    edit_prompt: bool = False          # allow editing inside prompt region?
    temperature: float = 0.7           # token sampling temperature (sub/ins)
    verbose: bool = True               # whether to show intermediate decoding traces
    time_independent: bool = True


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
    prev_out: Optional[dict] = None,      # <-- pass prior step's model outputs
    reuse_prev: bool = False,             # <-- if True, reuse prev_out instead of forward()
) -> Tuple[torch.Tensor, bool, dict, dict]:
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

    del_rate_h = out["del_rate_hat"]      # [1, T]
    sub_rate_h = out["sub_rate_hat"]      # [1, T]
    ins_rate_h = out["ins_rate_hat"]      # [1, T]
    sub_logits = out["sub_logits"]        # [1, T, V]
    ins_logits = out["ins_logits"]        # [1, T, V]

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
            ins_rate[:, :prompt_len_clamped-1] = 0.0

    # Combined "edit" (delete or substitute) event
    comb_rate = (del_rate + sub_rate).squeeze(0)                     # [T]
    comb_fire = _bernoulli_from_rate(comb_rate, cfg.tau).bool()      # [T]

    # If an edit fires at i, choose deletion with prob λ_del/(λ_del+λ_sub)
    p_del = (del_rate.squeeze(0) / (comb_rate + 1e-8)).clamp(0, 1)   # [T]
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire        # [T]
    choose_sub = comb_fire & (~choose_del)                           # [T]

    # Insertions (right of token i)
    ins_fire = _bernoulli_from_rate(ins_rate.squeeze(0), cfg.tau).bool()  # [T]

    # Token draws (algorithmic, not viz-only)
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

    # --- viz-only per-position labels (for trace/GIF) ---
    _before_ops: Annotated[List[str], "viz-only"] = []  # per 'before' position: DEL/SUB/KEEP
    _after_ops:  Annotated[List[str], "viz-only"] = []  # per 'after' token aligned to new_ids: INS/SUB/KEEP

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
        _ops_strs: Annotated[List[str], "viz-only"] = []
        for i in range(T):
            if choose_del[i]:
                _ops_strs.append(f"DEL@{i}:{_tok_str(int(x[i]))}")
            elif choose_sub[i]:
                _ops_strs.append(f"SUB@{i}:{_tok_str(int(x[i]))}->{_tok_str(sub_samples[i])}")
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
        "before_ops": _before_ops,   # viz-only labels
        "after_ops": _after_ops,     # viz-only labels
        # below are algorithmic signals copied for visualization/analysis
        "choose_del": [bool(v) for v in choose_del.tolist()],
        "choose_sub": [bool(v) for v in choose_sub.tolist()],
        "ins_fire":   [bool(v) for v in ins_fire.tolist()],
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
) -> Tuple[str, dict]:
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
            raise ValueError("Tokenizer must define mask_token_id when --mask_length > 0.")
        ids = ids + [tokenizer.mask_token_id] * args.mask_length

    x = torch.tensor(ids, dtype=torch.long, device=model.device)

    sched = LinearKappaScheduler()
    tau = cfg.tau
    steps = math.ceil(1.0 / max(tau, 1e-9))

    _trace: Annotated[dict, "viz-only: full decode trace for GIF/inspection"] = {
        "steps": [],
        "init": {"t": 0.0, "x_ids": [int(i) for i in x.tolist()], "prompt_len": int(prompt_len)},
        "end_t": 0.0,
    }

    # Local-only reuse: if previous iteration had no edits, reuse its forward.
    prev_out: Optional[dict] = None
    prev_had_edits = True  # first iteration must run a forward

    t = 0.0
    for _ in range(steps):
        # We can reuse prev_out only if the model is declared time-independent
        # and the previous step had NO edits (sequence unchanged).
        reuse_prev = (cfg.time_independent and not prev_had_edits and (prev_out is not None))

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
    font_size: int = 15,
    line_spacing: int = 12,
    frame_ms: int = 250,
    clean_final_ms: int = 10000,  # final clean frame duration
    max_width: int = 1400,
    max_height: int = 1600,  # kept as a cap
    margin: int = 32,
    title_color=(80, 80, 80),
    text_color=(0, 0, 0),
    mask_color=(150, 150, 150),
    sub_color=(0, 0, 200),
    ins_color=(200, 0, 0),
    del_strike_color=(120, 120, 120),
    events_color=(30, 30, 30),
    box_color=(120, 120, 120),
    bg_color=(255, 255, 255),
):
    """
    Same logic as your original, but uses a single canvas height across all frames:
    H = min(max_height, max(required_height_per_frame)).
    """
    from PIL import Image, ImageDraw, ImageFont
    import unicodedata

    # --- font ---
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # --- helpers ---
    def _sanitize_token(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("Ċ", "\n").replace("▁", " ").replace("Ġ", " ")
        s = s.replace("\t", "    ")
        s = s.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
        return s

    def _tok_str(tok_id: int) -> str:
        try:
            s = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
            s = s if s.strip() else f"<{int(tok_id)}>"
        except Exception:
            s = f"<{int(tok_id)}>"
        return s.replace("\n", "\\n")

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

    # wrapping
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
                        lines.append(cur)
                        cur, cur_w = [], 0
                if j != len(parts) - 1:
                    lines.append(cur)
                    cur, cur_w = [], 0
        if cur:
            lines.append(cur)
        return lines

    def _draw_dashed_rectangle(
        draw, xy, dash=8, gap=6, width=2, outline=(120, 120, 120)
    ):
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

    # ---------- PASS 1: measure required heights per frame ----------
    # We collect all wrapping/ops data first, so we can pick a uniform H.
    measurement_payload = (
        []
    )  # list of dicts with wrapped_lines, ops_lines, and step ref

    for step_idx, st in enumerate([None] + trace["steps"]):
        # build augmented sequence with injected deleted tokens
        if step_idx == 0:
            augmented_ids = trace["init"]["x_ids"]
            deleted_flags = [False] * len(augmented_ids)
        else:
            after_ids = list(st["x_after_ids"])
            before_ids = st["x_before_ids"]
            choose_del = st["choose_del"]
            deleted_positions = [i for i, d in enumerate(choose_del) if d]

            augmented_ids = list(after_ids)
            deleted_flags = [False] * len(after_ids)
            for i in sorted(deleted_positions, reverse=True):
                augmented_ids.insert(i, before_ids[i])
                deleted_flags.insert(i, True)

        tokens = tokenizer.convert_ids_to_tokens(augmented_ids)
        wrapped_lines = _wrap_tokens_with_index(tokens, deleted_flags)
        ops_lines = _ops_lines_for_step(st)

        # required height for this frame (title + body + spacer + ops box)
        body_h = len(wrapped_lines) * (font_size + line_spacing)
        ops_h = (
            len(ops_lines) * (font_size + line_spacing) + font_size + 20
        )  # matches draw math
        required_h = (
            margin + (font_size + line_spacing) + body_h + 20 + ops_h
        )  # no extra bottom margin

        measurement_payload.append(
            {
                "step_idx": step_idx,
                "st": st,
                "augmented_ids": augmented_ids,
                "tokens": tokens,
                "wrapped_lines": wrapped_lines,
                "ops_lines": ops_lines,
                "required_h": required_h,
            }
        )

    # Measure clean final frame as well (no events box)
    final_text_ids = (
        trace["steps"][-1]["x_after_ids"] if trace["steps"] else trace["init"]["x_ids"]
    )
    final_tokens = tokenizer.convert_ids_to_tokens(final_text_ids)
    wrapped_clean = _wrap_tokens_with_index(final_tokens, [False] * len(final_tokens))
    clean_body_h = len(wrapped_clean) * (font_size + line_spacing)
    clean_required_h = (
        margin + (font_size + line_spacing) + clean_body_h
    )  # title + body

    # Pick a single canvas height: max across all frames, capped by max_height
    max_required_h = (
        max([p["required_h"] for p in measurement_payload] + [clean_required_h]) + 20
    )
    H = min(max_required_h, max_height)
    W = max_width

    # ---------- PASS 2: render with uniform H ----------
    from PIL import Image

    frames = []

    for p in measurement_payload:
        step_idx = p["step_idx"]
        st = p["st"]
        tokens = p["tokens"]
        wrapped_lines = p["wrapped_lines"]
        augmented_ids = p["augmented_ids"]
        ops_lines = p["ops_lines"]

        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

        # title
        title = "initial state" if step_idx == 0 else f"t = {st['t']:.3f}"
        y_title = margin
        draw.text((margin, y_title), title, fill=title_color, font=font)
        y = y_title + font_size + line_spacing

        # body
        for line in wrapped_lines:
            x = margin
            for seg_text, tok_idx, is_deleted in line:
                if seg_text == "":
                    continue
                tok_id = int(augmented_ids[tok_idx])
                tok_str_exact = tokens[tok_idx]
                if is_deleted:
                    strike_color = (
                        mask_color
                        if (tok_id in MASK_IDS or tok_str_exact in MASK_STRINGS)
                        else del_strike_color
                    )
                    strike = "".join(ch + "\u0336" for ch in seg_text)
                    draw.text((x, y), strike, fill=strike_color, font=font)
                    x += tmp_draw.textlength(strike, font=font)
                else:
                    color = (
                        mask_color
                        if (tok_id in MASK_IDS or tok_str_exact in MASK_STRINGS)
                        else text_color
                    )
                    if step_idx != 0 and tok_idx < len(st["after_ops"]):
                        op = st["after_ops"][tok_idx]
                        if op == "INS":
                            color = ins_color
                        elif op == "SUB":
                            color = sub_color
                    draw.text((x, y), seg_text, fill=color, font=font)
                    x += tmp_draw.textlength(seg_text, font=font)
            y += font_size + line_spacing

        # events box
        y += 20
        x0, y0 = margin, y
        x1 = W - margin
        ops_box_h = len(ops_lines) * (font_size + line_spacing) + font_size + 20
        y1 = y0 + ops_box_h
        _draw_dashed_rectangle(draw, (x0, y0, x1, y1), outline=box_color)
        draw.text((x0 + 10, y0 + 10), "events", fill=events_color, font=font)
        yy = y0 + font_size + 20
        for line in ops_lines:
            draw.text((x0 + 10, yy), line, fill=events_color, font=font)
            yy += font_size + line_spacing

        frames.append(img)

    # clean final frame (no events box)
    clean_img = Image.new("RGB", (W, H), bg_color)
    clean_draw = ImageDraw.Draw(clean_img)
    clean_draw.text((margin, margin), "final text", fill=title_color, font=font)
    y = margin + font_size + line_spacing
    for line in wrapped_clean:
        x = margin
        for seg_text, _, _ in line:
            clean_draw.text((x, y), seg_text, fill=text_color, font=font)
            x += tmp_draw.textlength(seg_text, font=font)
        y += font_size + line_spacing
    frames.append(clean_img)

    # save gif
    durations = [frame_ms] * (len(frames) - 1) + [clean_final_ms]
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
        time_independent: Annotated[bool, "Whether model is conditioned on time step"] = True

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

        # Visualization
        make_gif: Annotated[bool, "Render a decoding trace GIF after generation."] = False
        gif_path: Annotated[Optional[str], "Output GIF path (default: decode_trace.gif)"] = None
        frame_ms: Annotated[int, "Per-frame duration in ms"] = 120
        # show_token_pieces: Annotated[bool,
        #     "True = color token pieces (blue=INS, orange=SUB, black=KEEP). "
        #     "False = show human-readable decoded text (no per-token colors)."] = True

    args = tyro.cli(ScriptArgs)

    cfg = GenCfg(
        tau=args.tau,
        seed=args.seed,
        edit_prompt=args.edit_prompt,
        temperature=args.temperature,
        verbose=args.verbose,
        time_independent=args.time_independent,
        # make_gif=args.make_gif,
        # gif_path=args.gif_path,
        # frame_ms=args.frame_ms,
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
