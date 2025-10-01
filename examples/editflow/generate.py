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
# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 python examples/editflow/generate.py --model_name_or_path "models/EditFlow-Dream-7B/tulu-3-sft-mixture-[mix]-opc-sft-stage2/checkpoint-1260"  --tau 0.1 --mask_length 256 --seed 7070  --prompt "write a romantic story" --make_gif

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Annotated

import tyro
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dllm.utils.schedulers import BaseKappaScheduler, LinearKappaScheduler

# Visualization deps
from PIL import Image, ImageDraw, ImageFont


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
    # Visualization (optional)
    make_gif: bool = False
    gif_path: Optional[str] = None     # e.g., "decode_trace.gif"
    frame_ms: int = 600                # per-frame duration


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
) -> Tuple[torch.Tensor, bool, dict]:
    """
    Single τ-leap step with deletion/substitution conflict resolution
    and right-insert policy.

    - BOS (index 0) is always protected from delete/substitute.
    - If cfg.edit_prompt is False, protect the prompt span [0, prompt_len):
        * del/sub disabled for i in [0, prompt_len)
        * insertion disabled for i in [0, prompt_len-1), but allowed at i=prompt_len-1
          (the last prompt token) and anywhere i >= prompt_len.

    Returns:
        x_next, any_edit, step_trace (dict with all intermediates for visualization)
    """
    device = x.device
    T = x.numel()

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
    w = sched.weight(tt)          # shape [1] or [1,1] depending on impl
    del_rate = del_rate_h * w
    sub_rate = sub_rate_h * w
    ins_rate = ins_rate_h * w

    # Clamp prompt_len within current T (robustness)
    prompt_len_clamped = int(max(1, min(prompt_len, T)))

    # Disable deleting and substituting BOS token
    del_rate[:, 0] = 0.0
    sub_rate[:, 0] = 0.0

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
    before_ops: List[str] = []  # per "before" position: DEL/SUB/KEEP
    after_ops: List[str] = []   # per "after" token aligned to new_ids: INS/SUB/KEEP

    for i in range(T):
        if choose_del[i]:
            before_ops.append("DEL")  # dropped below
        elif choose_sub[i]:
            before_ops.append("SUB")
            new_tok = sub_samples[i]
            new_ids.append(int(new_tok))
            after_ops.append("SUB")
        else:
            before_ops.append("KEEP")
            new_ids.append(int(x[i].item()))
            after_ops.append("KEEP")
        if ins_samples[i] is not None:
            new_ids.append(int(ins_samples[i]))
            after_ops.append("INS")

    # Verbose console trace (optional)
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
        print("[time]", f"{t:.4f}")
        print("[events]", "; ".join(ops))
        print("[decode]\n", tokenizer.decode(new_ids, skip_special_tokens=True))
        print()

    x_next = torch.tensor(new_ids, dtype=torch.long, device=device)
    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())

    # Step trace payload for visualization
    step_trace = {
        "t": float(t),
        "x_before_ids": [int(i) for i in x.tolist()],
        "x_after_ids": [int(i) for i in new_ids],
        "before_ops": before_ops,  # len = len(x_before_ids), values in {"DEL","SUB","KEEP"}
        "after_ops": after_ops,    # len = len(x_after_ids), values in {"INS","SUB","KEEP"}
        "choose_del": [bool(v) for v in choose_del.tolist()],
        "choose_sub": [bool(v) for v in choose_sub.tolist()],
        "ins_fire":   [bool(v) for v in ins_fire.tolist()],
        "sub_samples": [int(s) if s is not None else None for s in sub_samples],
        "ins_samples": [int(s) if s is not None else None for s in ins_samples],
        "prompt_len": prompt_len_clamped,
    }

    return x_next, any_edit, step_trace


# -------------------------------- top-level generate -------------------------------

@torch.no_grad()
def generate_editflow_minimal(model, tokenizer, args, cfg: GenCfg):
    """
    Returns:
        final_text, trace
    where trace is:
        {
          "steps": [ step_trace_0, step_trace_1, ... ],
          "init":  { "t": 0.0, "x_ids": [...], "prompt_len": int },
          "end_t": float
        }
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
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt, add_special_tokens=False)
        ids = [bos] + enc["input_ids"]  # ALWAYS prefix BOS

    prompt_len = len(ids)

    if args.mask_length:
        if getattr(tokenizer, "mask_token_id", None) is None:
            raise ValueError("Tokenizer must define mask_token_id when --mask_length > 0.")
        ids = ids + [tokenizer.mask_token_id] * args.mask_length

    x = torch.tensor(ids, dtype=torch.long, device=model.device)

    sched = LinearKappaScheduler()
    tau = cfg.tau
    steps = math.ceil(1.0 / max(tau, 1e-9))

    trace = {
        "steps": [],
        "init": {"t": 0.0, "x_ids": [int(i) for i in x.tolist()], "prompt_len": int(prompt_len)},
        "end_t": 0.0,
    }

    t = 0.0
    for _ in range(steps):
        x, edited, step_trace = tau_leap_step_minimal(
            x=x,
            model=model,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            t=t,
            sched=sched,
            cfg=cfg
        )
        trace["steps"].append(step_trace)
        t = min(1.0, t + tau)
        if t >= 1.0 - args.time_epsilon:
            break

    trace["end_t"] = float(t)

    final_text = tokenizer.decode(x.tolist(), skip_special_tokens=True)
    print("[final]")
    return final_text, trace


# ------------------------------ Visualization (NEW) ------------------------------
# Diffusion-style consecutive output: only show the CURRENT output per frame.
# ------------------ Visualization (sanitized, masks stripped) ------------------
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
import re
import unicodedata

def render_consecutive_trace_gif(
    trace: dict,
    tokenizer: PreTrainedTokenizer,
    out_path: str = "decode_trace.gif",
    font_size: int = 30,
    line_spacing: int = 12,
    frame_ms: int = 120,          # per-frame duration for all but last
    final_hold_ms: int = 10_000,  # hold the final frame for 10s
    max_width: int = 1600,
    margin: int = 32,
    title_color=(80, 80, 80),
    text_color=(0, 0, 0),
    events_color=(30, 30, 30),
    box_color=(120, 120, 120),
    bg_color=(255, 255, 255),
):
    """
    Diffusion-style GIF of the *current* decoded sequence (PROMPT + RESPONSE), keeping
    special tokens (e.g., <|im_start|>, <|im_end|>) and stripping only mask tokens.
    Adds a bottom dashed 'events' box listing DEL/SUB/INS for the current frame.
    Uniform canvas height across frames; final frame held for 10s.
    """

    # ---- font ----
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # ---- helpers ----
    # Only strip mask IDs; keep other specials
    def _mask_ids(tok: PreTrainedTokenizer) -> set:
        s = set()
        v = getattr(tok, "mask_token_id", None)
        if v is not None:
            s.add(int(v))
        return s
    MASK_IDS = _mask_ids(tokenizer)

    mask_literals = {"<mask>", "[MASK]", "[mask]", "<MASK>", "<|mask|>", "|mask|", "MASK", "mask"}
    mt = getattr(tokenizer, "mask_token", None)
    if mt:
        mask_literals.add(str(mt))
    MASK_RE = re.compile(
        r"(?:▁|Ġ)?(?:" + "|".join(re.escape(m) for m in sorted(mask_literals, key=len, reverse=True)) + r")",
        flags=re.IGNORECASE
    )

    def _remove_only_masks(ids: List[int]) -> List[int]:
        return [int(i) for i in ids if int(i) not in MASK_IDS]

    def _decode_keep_specials(ids: List[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    BYTE_FALLBACK_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")

    def _sanitize(text: str) -> str:
        """Normalize artifacts while preserving spaces/indentation and special tokens."""
        if not text:
            return ""
        s = unicodedata.normalize("NFKC", text)

        # Strip literal mask strings (before other cleanup)
        s = MASK_RE.sub("", s)

        # Replace BPE/SPM markers -> real whitespace/newline
        s = s.replace("Ċ", "\n")   # newline
        s = s.replace("▁", " ")    # spm space marker
        s = s.replace("Ġ", " ")    # gpt2 bpe space marker

        # Map byte-fallback tokens (keep LF and space; drop others)
        def _byte_sub(m):
            b = int(m.group(1), 16)
            if b == 0x0A: return "\n"
            if b == 0x20: return " "
            return ""
        s = BYTE_FALLBACK_RE.sub(_byte_sub, s)

        # Normalize whitespace *types* but NOT counts
        s = s.replace("\t", "    ")
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")

        # Do not collapse spaces; trim trailing spaces only
        s = "\n".join(line.rstrip() for line in s.splitlines())
        return s

    # Indentation-safe wrappers (preserve multiple spaces)
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

    # Build ops list lines for a given step
    def _tok_str(tok_id: int) -> str:
        try:
            s = tokenizer.decode([int(tok_id)], skip_special_tokens=False)
            s = s if s.strip() else f"<{int(tok_id)}>"
        except Exception:
            s = f"<{int(tok_id)}>"
        # keep special tokens; sanitize common artifacts lightly to avoid noise in ops
        s = s.replace("\n", "\\n")
        return s

    def _ops_lines_for_step(st: Optional[dict]) -> List[str]:
        if st is None:
            return ["(no events)"]
        lines: List[str] = []
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
                lines.append(f"SUB@{i}:{_tok_str(int(x_before[i]))}->{_tok_str(int(sub_samples[i]))}")
            if ins_samples[i] is not None:
                lines.append(f"INS@{i}->{i+1}:{_tok_str(int(ins_samples[i]))}")
        if not lines:
            lines.append("(no events)")
        return lines

    # Dashed rectangle helper
    def _draw_dashed_rectangle(draw: ImageDraw.ImageDraw, xy, dash=6, gap=6, width=2, fill=None, outline=(120,120,120)):
        x0, y0, x1, y1 = xy
        # top
        x = x0
        while x < x1:
            x2 = min(x + dash, x1)
            draw.line([(x, y0), (x2, y0)], fill=outline, width=width)
            x += dash + gap
        # bottom
        x = x0
        while x < x1:
            x2 = min(x + dash, x1)
            draw.line([(x, y1), (x2, y1)], fill=outline, width=width)
            x += dash + gap
        # left
        y = y0
        while y < y1:
            y2 = min(y + dash, y1)
            draw.line([(x0, y), (x0, y2)], fill=outline, width=width)
            y += dash + gap
        # right
        y = y0
        while y < y1:
            y2 = min(y + dash, y1)
            draw.line([(x1, y), (x1, y2)], fill=outline, width=width)
            y += dash + gap

    # --- measure context ---
    tmp_img = Image.new("RGB", (10, 10), bg_color)
    tmp_draw = ImageDraw.Draw(tmp_img)
    text_width_budget = max_width - 2 * margin

    # --- collect all frames' body + ops first (PROMPT+RESPONSE) ---
    frames_payload: List[dict] = []

    # Initial frame from init (full sequence)
    init_ids_full = _remove_only_masks(trace["init"]["x_ids"])
    init_body = _sanitize(_decode_keep_specials(init_ids_full))
    frames_payload.append({
        "t": None,
        "del_count": 0,
        "body_lines": _wrap_text(tmp_draw, init_body, text_width_budget),
        "ops_lines": _ops_lines_for_step(None),
    })

    # Steps
    for st in trace["steps"]:
        after_ids_full = _remove_only_masks(st["x_after_ids"])
        body = _sanitize(_decode_keep_specials(after_ids_full))
        frames_payload.append({
            "t": float(st["t"]),
            "del_count": sum(1 for op in st["before_ops"] if op == "DEL"),
            "body_lines": _wrap_text(tmp_draw, body, text_width_budget),
            "ops_lines": _wrap_text(tmp_draw, "  • " + "  • ".join(_ops_lines_for_step(st)), text_width_budget),
            # Using bullets and wrap to avoid over-wide op rows
        })

    # --- compute uniform canvas height (max body + max ops) ---
    max_body_lines = max(len(f["body_lines"]) for f in frames_payload)
    # For the ops box: header + lines + padding
    # Re-wrap ops without the bullet-join to also support many short lines:
    def _measure_ops_lines(f):
        return len(f["ops_lines"])
    max_ops_lines = max(_measure_ops_lines(f) for f in frames_payload)

    title_block = font_size + line_spacing
    body_block  = max_body_lines * (font_size + line_spacing)

    # Ops box metrics
    ops_header = font_size  # "events" label
    ops_pad = 10
    ops_lines_block = max_ops_lines * (font_size + line_spacing)
    ops_box_height = ops_pad + ops_header + line_spacing//2 + ops_lines_block + ops_pad

    H = margin + title_block + body_block + line_spacing + ops_box_height + margin
    W = max_width

    # --- render frames ---
    frames: List[Image.Image] = []
    for f in frames_payload:
        img = Image.new("RGB", (W, H), bg_color)
        draw = ImageDraw.Draw(img)

        # Title
        title = ("initial state" if f["t"] is None else f"t = {f['t']:.3f}") + (f"   \u232b{f['del_count']}" if f["del_count"] > 0 else "")
        draw.text((margin, margin), title, fill=title_color, font=font)

        # Body text
        y = margin + title_block
        for line in f["body_lines"]:
            draw.text((margin, y), line, fill=text_color, font=font)
            y += font_size + line_spacing

        # Events dashed box
        y += line_spacing
        x0, y0 = margin, y
        x1, y1 = W - margin, y + ops_box_height
        _draw_dashed_rectangle(draw, (x0, y0, x1, y1), dash=8, gap=6, width=2, outline=box_color)

        # "events" label inside the box
        label_x = x0 + ops_pad
        label_y = y0 + ops_pad
        draw.text((label_x, label_y), "events", fill=events_color, font=font)

        # Ops lines
        yy = label_y + ops_header + line_spacing//2
        for line in f["ops_lines"]:
            draw.text((label_x, yy), line, fill=events_color, font=font)
            yy += font_size + line_spacing

        frames.append(img)

    # --- save GIF with per-frame durations; final frame held 10s ---
    if len(frames) == 1:
        frames[0].save(out_path)
    else:
        durations = [frame_ms] * (len(frames) - 1) + [final_hold_ms]
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

        prompt: Annotated[Optional[str], "Text prompt. If None, start from BOS alone."] = None
        # Boolean flag: tyro exposes --edit-prompt / --no-edit-prompt automatically for bools
        edit_prompt: Annotated[bool,
            "Allow delete/substitute and insertions in the prompt region (BOS+prompt)."] = False

        # Generation-related args
        tau: Annotated[float, "τ-leap size"] = 0.02
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
        make_gif=args.make_gif,
        gif_path=args.gif_path,
        frame_ms=args.frame_ms,
    )

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    final_text, trace = generate_editflow_minimal(model, tokenizer, args, cfg)
    print(final_text)

    if cfg.make_gif:
        out = cfg.gif_path or "decode_trace.gif"
        path = render_consecutive_trace_gif(
            trace,
            tokenizer,
            out_path=out,
            frame_ms=cfg.frame_ms,
        )
        print(f"[gif saved] {path}")


if __name__ == "__main__":
    main()
