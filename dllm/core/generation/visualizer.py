from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence, Optional

import torch
from tqdm import tqdm

from transformers import PreTrainedTokenizer



@dataclass
class BaseVisualizer(ABC):
    tokenizer: PreTrainedTokenizer

    @abstractmethod
    def visualize(self, history: list[torch.Tensor, list], **kwargs):
        raise NotImplementedError

@dataclass
class VideoVisualizer(BaseVisualizer):

    def visualize(
        self, 
        history: list[torch.Tensor, list], 
        output_path: str = "visualization.gif", 
        **kwargs
    ):
        raise NotImplementedError

# @dataclass
# class TerminalVisualizer(BaseVisualizer):
#     def visualize(
#         self,
#         history: list[torch.Tensor],  # list of tokens per step: [T] or [B,T]
#         fps: int = 6,
#         rich: bool = False,
#         title: str = "dllm",
#         max_chars: int = 4000,
#         every_n_steps: int = 1,          # re-render frequency (perf knob)
#         show_header: bool = True,
#         skip_special_tokens: bool = False,  # NEW ARGUMENT
#     ) -> None:
#         """
#         Visualize a masked-diffusion decoding trajectory stored in `history`.

#         Args:
#             history: Sequence of token tensors for each step. Each item is [T] or [B,T].
#             fps: Frames per second for the live UI (Rich) or sleep cadence for tqdm fallback.
#             title: Header title.
#             max_chars: Cap on rendered characters to keep terminal snappy.
#             every_n_steps: Only redraw text every N steps (progress still updates every step).
#             show_header: Show the magenta header bar (Rich path).
#             skip_special_tokens: Whether to skip special/pad/eos tokens when rendering (default: False).

#         Notes:
#             - Masked positions are detected via `self.tokenizer.mask_token_id`.
#             - Special tokens are determined via `self.tokenizer.all_special_ids`.
#             - All layout, styling, and progress are encapsulated here.
#         """

#         try:
#             from rich.console import Console
#             from rich.live import Live
#             from rich.text import Text
#             from rich.panel import Panel
#             from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
#             from rich.layout import Layout
#             _RICH = True
#         except Exception:
#             _RICH = False

#         if self.tokenizer is None:
#             raise ValueError("TerminalVisualizer.tokenizer must be set to a valid tokenizer.")

#         tokenizer = self.tokenizer
#         specials: set[int] = set(getattr(tokenizer, "all_special_ids", []) or [])
#         mask_token_id: Optional[int] = getattr(tokenizer, "mask_token_id", None)
#         pad_token_id: Optional[int] = getattr(tokenizer, "pad_token_id", None)
#         eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

#         def _first_item(x: torch.Tensor) -> torch.Tensor:
#             return x[0] if x.dim() > 1 else x

#         def _count_masks(toks: torch.Tensor) -> int:
#             if mask_token_id is None:
#                 return 0
#             t = _first_item(toks)
#             return int((t == mask_token_id).sum().item())

#         def _decode_token_id(tid: int) -> Optional[str]:
#             if skip_special_tokens and tid in specials:
#                 return None
#             try:
#                 s = tokenizer.convert_ids_to_tokens([tid])[0]
#                 if not s:
#                     s = tokenizer.decode([tid], skip_special_tokens=skip_special_tokens)
#             except Exception:
#                 try:
#                     s = tokenizer.decode([tid], skip_special_tokens=skip_special_tokens)
#                 except Exception:
#                     s = None
#             return s

#         def _format_text(tokens: torch.Tensor, step: int, total: int) -> Text:
#             txt = Text()
#             t = _first_item(tokens).long()
#             T = int(t.numel())
#             cur = 0

#             for i in range(T):
#                 tid = int(t[i].item())

#                 # treat mask token as masked cell
#                 if mask_token_id is not None and tid == mask_token_id:
#                     style = "bold black on yellow" if (step % 2 == 0) else "bold yellow on red"
#                     token_str = "[MASK]"
#                     if max_chars and cur + len(token_str) > max_chars:
#                         txt.append(" …", style="dim"); break
#                     txt.append(token_str, style=style)
#                     cur += len(token_str)
#                     continue

#                 # optionally skip pad/eos/specials
#                 if skip_special_tokens:
#                     if tid in specials or (pad_token_id is not None and tid == pad_token_id) or (eos_token_id is not None and tid == eos_token_id):
#                         continue

#                 s = _decode_token_id(tid)
#                 if not s:
#                     continue
#                 s = s.replace("\r", " ").replace("\n", " ")
#                 if max_chars and cur + len(s) > max_chars:
#                     txt.append(" …", style="dim"); break
#                 txt.append(s, style="green")
#                 cur += len(s)

#             if tokens.dim() == 2 and tokens.size(0) > 1:
#                 txt.append(f"  [dim](+{tokens.size(0)-1} more in batch)[/dim]")
#             return txt

#         total_steps = len(history)
#         sleep_s = 0.0 if fps <= 0 else 1.0 / float(fps)

#         # ---------- tqdm fallback ----------
#         if not _RICH or not rich:
#             pbar = tqdm(total=total_steps, desc="Diffusion", leave=True)
#             for i, toks in enumerate(history, start=1):
#                 pbar.update(1)
#                 pbar.set_postfix({
#                     "masks": _count_masks(toks),
#                     "pct": f"{int(100 * i / max(total_steps, 1))}%",
#                 })
#                 if sleep_s > 0:
#                     time.sleep(sleep_s)
#             pbar.close()
#             print("\n✨ Generation complete!\n")
#             try:
#                 final_seq = _first_item(history[-1])
#                 final_text = tokenizer.decode(final_seq, skip_special_tokens=skip_special_tokens)
#                 if final_text:
#                     print(final_text)
#             except Exception:
#                 pass
#             return

#         # ---------- rich live UI ----------
#         console = Console(force_terminal=True, color_system="truecolor")
#         layout = Layout()
#         layout.split_column(
#             Layout(name="header", size=3) if show_header else Layout(name="header", size=0),
#             Layout(name="text", ratio=1),
#             Layout(name="progress", size=3),
#         )

#         progress = Progress(
#             SpinnerColumn(),
#             TextColumn("[bold blue]Diffusion"),
#             BarColumn(),
#             MofNCompleteColumn(),
#             TextColumn("•"),
#             TextColumn("[cyan]Masks: {task.fields[masks]}"),
#             TextColumn("•"),
#             TextColumn("[magenta]{task.fields[pct]:>4s}"),
#             TimeRemainingColumn(),
#             expand=True,
#         )

#         init_masks = _count_masks(history[0]) if history else 0
#         task_id = progress.add_task("Generating", total=total_steps, masks=init_masks, pct="0%")

#         with Live(layout, console=console, refresh_per_second=max(1, fps)):
#             for step_idx, toks in enumerate(history, start=1):
#                 if show_header:
#                     header = Text(title, style="bold magenta", justify="center")
#                     layout["header"].update(Panel(header, border_style="bright_blue"))

#                 masks_remaining = _count_masks(toks)
#                 pct = f"{int(100 * step_idx / max(total_steps, 1))}%"
#                 progress.update(task_id, advance=1, masks=masks_remaining, pct=pct)

#                 if every_n_steps <= 1 or (step_idx % every_n_steps == 0) or step_idx in (1, total_steps):
#                     text_rich = _format_text(toks, step_idx, total_steps)
#                     layout["text"].update(
#                         Panel(
#                             text_rich if text_rich.plain else Text("[dim]— no tokens —[/dim]"),
#                             title="[bold]Generated Text",
#                             subtitle=f"[dim]Step {step_idx}/{total_steps}[/dim]",
#                             border_style="cyan",
#                             padding=(1, 1),
#                         )
#                     )

#                 layout["progress"].update(Panel(progress))
#                 if sleep_s > 0:
#                     time.sleep(sleep_s)

#         console.print("\n[bold green]✨ Generation complete![/bold green]\n")
#         try:
#             final_seq = _first_item(history[-1])
#             final_text = tokenizer.decode(final_seq, skip_special_tokens=skip_special_tokens)
#             console.print(Panel(final_text if final_text else "[dim]— no decodable text —[/dim]",
#                                 title="[bold]Final Generated Text", border_style="green", padding=(1, 2)))
#         except Exception:
#             pass

@dataclass
class TerminalVisualizer(BaseVisualizer):
    def visualize(
        self,
        history: list[torch.Tensor],  # list of tokens per step: [T] or [B,T]
        fps: int = 16,
        rich: bool = True,
        title: str = "dllm",
        max_chars: int = None,
        every_n_steps: int = 1,          # re-render frequency (perf knob)
        show_header: bool = True,
        skip_special_tokens: bool = False,  # NEW ARGUMENT
    ) -> None:
        """
        Visualize a masked-diffusion decoding trajectory stored in `history`.

        Args:
            history: Sequence of token tensors for each step. Each item is [T] or [B,T].
            fps: Frames per second for the live UI (Rich) or sleep cadence for tqdm fallback.
            title: Header title.
            max_chars: Cap on rendered characters to keep terminal snappy.
            every_n_steps: Only redraw text every N steps (progress still updates every step).
            show_header: Show the magenta header bar (Rich path).
            skip_special_tokens: Whether to skip special/pad/eos tokens when rendering (default: False).

        Notes:
            - Masked positions are detected via `self.tokenizer.mask_token_id`.
            - Special tokens are determined via `self.tokenizer.all_special_ids`.
            - All layout, styling, and progress are encapsulated here.
        """
        # --------- imports & env checks ----------
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.text import Text
            from rich.panel import Panel
            from rich.progress import (
                Progress, BarColumn, TextColumn, TimeRemainingColumn,
                MofNCompleteColumn, SpinnerColumn
            )
            from rich.layout import Layout
            _RICH_IMPORTED = True
        except Exception:
            _RICH_IMPORTED = False

        try:
            from tqdm import tqdm
            _TQDM_IMPORTED = True
        except Exception:
            _TQDM_IMPORTED = False

        if self.tokenizer is None:
            raise ValueError("TerminalVisualizer.tokenizer must be set to a valid tokenizer.")

        tokenizer = self.tokenizer
        specials: set[int] = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._specials = specials  # store for helpers
        self._mask_token_id: Optional[int] = getattr(tokenizer, "mask_token_id", None)
        self._pad_token_id: Optional[int] = getattr(tokenizer, "pad_token_id", None)
        self._eos_token_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)

        # --------- helpers inside class scope ----------
        # (keep everything inside this class as requested)

        # throttle settings
        sleep_s = 0.0 if fps <= 0 else 1.0 / float(max(1, fps))
        total_steps = len(history)
        every_n_steps = max(1, int(every_n_steps))

        # decode final text up-front (used after render)
        final_text = self._detok(history[-1], skip_special_tokens=skip_special_tokens)
        final_text = self._truncate(final_text, max_chars)

        # choose rich or tqdm
        use_rich = bool(rich and _RICH_IMPORTED)

        if not use_rich or not _RICH_IMPORTED:
            # ---------- tqdm fallback ----------
            if not _TQDM_IMPORTED:
                for i, toks in enumerate(history, start=1):
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                print("\n✨ Generation complete!\n")
                print(final_text)
                return

            pbar = tqdm(total=total_steps, desc="Diffusion", leave=True)
            for i, toks in enumerate(history, start=1):
                pbar.update(1)
                pbar.set_postfix({
                    "masks": self._count_masks(toks),
                    "pct": f"{int(100 * i / max(total_steps, 1))}%",
                })
                if sleep_s > 0:
                    time.sleep(sleep_s)
            pbar.close()
            print("\n✨ Generation complete!\n")
            if final_text:
                print(final_text)
            return

        # ---------- rich live UI ----------
        console = Console(force_terminal=True, color_system="truecolor")
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3) if show_header else Layout(name="header", size=0),
            Layout(name="text", ratio=1),
            Layout(name="progress", size=3),
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Diffusion"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]Masks: {task.fields[masks]}"),
            TextColumn("•"),
            TextColumn("[magenta]{task.fields[pct]:>4s}"),
            TimeRemainingColumn(),
            expand=True,
        )

        init_masks = self._count_masks(history[0]) if history else 0
        task_id = progress.add_task("Generating", total=total_steps, masks=init_masks, pct="0%")

        with Live(layout, console=console, refresh_per_second=max(1, fps)):
            for step_idx, toks in enumerate(history, start=1):
                if show_header:
                    header = Text(title, style="bold magenta", justify="center")
                    layout["header"].update(Panel(header, border_style="bright_blue"))

                # progress bar
                masks_remaining = self._count_masks(toks)
                pct = f"{int(100 * step_idx / max(total_steps, 1))}%"
                progress.update(task_id, advance=1, masks=masks_remaining, pct=pct)

                # text panel: decode whole sequence (avoids Ġ/Ċ artifacts)
                if every_n_steps <= 1 or (step_idx % every_n_steps == 0) or step_idx in (1, total_steps):
                    text_str = self._detok(toks, skip_special_tokens=skip_special_tokens)
                    text_str = self._truncate(text_str, max_chars)
                    text_rich = Text.from_ansi(text_str) if text_str else Text("")
                    layout["text"].update(
                        Panel(
                            text_rich if text_rich.plain else Text("[dim]— no tokens —[/dim]"),
                            title="[bold]Generated Text",
                            subtitle=f"[dim]Step {step_idx}/{total_steps}[/dim]",
                            border_style="cyan",
                            padding=(1, 1),
                        )
                    )

                layout["progress"].update(Panel(progress))
                if sleep_s > 0:
                    time.sleep(sleep_s)

        console.print("\n[bold green]✨ Generation complete![/bold green]\n")
        console.print(
            Panel(
                final_text if final_text else "[dim]— no decodable text —[/dim]",
                title="[bold]Final Generated Text",
                border_style="green",
                padding=(1, 2),
            )
        )

    # ======================== helpers (kept inside class) ========================

    def _has_tty(self) -> bool:
        return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")

    def _first_item(self, x: torch.Tensor) -> torch.Tensor:
        return x[0] if x.dim() > 1 else x

    def _count_masks(self, toks: torch.Tensor) -> int:
        if getattr(self, "_mask_token_id", None) is None:
            return 0
        t = self._first_item(toks)
        return int((t == self._mask_token_id).sum().item())

    def _detok(self, ids_or_tensor, *, skip_special_tokens: bool) -> str:
        """
        Robust detokenize for list[int] / torch.Tensor([T]) / torch.Tensor([B,T]).
        Decode the whole sequence to avoid byte-level artifacts like Ġ/Ċ.
        """
        tokenizer = self.tokenizer
        # normalize to python list[int]
        if isinstance(ids_or_tensor, torch.Tensor):
            t = self._first_item(ids_or_tensor).long()
            ids = t.tolist()
        elif isinstance(ids_or_tensor, (list, tuple)):
            ids = list(ids_or_tensor)
        else:
            # unknown type
            return ""

        # Optionally drop specials/pad/eos *before* decode if desired
        if skip_special_tokens:
            keep = []
            specials = getattr(self, "_specials", set())
            pad_id = getattr(self, "_pad_token_id", None)
            eos_id = getattr(self, "_eos_token_id", None)
            for tid in ids:
                if tid in specials:
                    continue
                if pad_id is not None and tid == pad_id:
                    continue
                if eos_id is not None and tid == eos_id:
                    continue
                keep.append(tid)
            ids = keep

        # Prefer tokenizer.decode (handles Ġ/Ċ, merges properly)
        text = ""
        try:
            if hasattr(tokenizer, "decode"):
                text = tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            else:
                # fallback: tokens -> string
                toks = tokenizer.convert_ids_to_tokens(ids)
                if hasattr(tokenizer, "convert_tokens_to_string"):
                    text = tokenizer.convert_tokens_to_string(toks)
                else:
                    text = " ".join(map(str, toks))
        except Exception:
            # extremely defensive fallback
            try:
                text = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                text = ""

        # sanitize control chars for terminal
        if text:
            text = text.replace("\r", "")
        return text

    def _truncate(self, s: str, max_chars: Optional[int]) -> str:
        if max_chars is None or (isinstance(max_chars, int) and max_chars < 0):
            return s
        return s[:max_chars]


if __name__ == "__main__":
    pass
