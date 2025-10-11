# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
Terminal visualization for RND1 generation.

This module provides real-time visualization of the diffusion denoising process,
showing token evolution and generation progress in the terminal using rich
formatting when available.
"""

import torch
from typing import Optional
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class TerminalVisualizer:
    """
    Rich-based visualization for diffusion process with live updates.

    Provides real-time visualization of the token denoising process during
    diffusion-based language generation, with colored highlighting of masked
    positions and progress tracking.
    """

    def __init__(self, tokenizer, show_visualization: bool = True):
        """
        Initialize the terminal visualizer.

        Args:
            tokenizer: The tokenizer for decoding tokens to text
            show_visualization: Whether to show visualization (requires rich)
        """
        self.tokenizer = tokenizer
        self.show_visualization = show_visualization and RICH_AVAILABLE
        if not RICH_AVAILABLE and show_visualization:
            print("Warning: Install 'rich' for better visualization. Falling back to simple progress bar.")
            self.show_visualization = False

        if self.show_visualization:
            self.console = Console()
            self.live = None
            self.progress = None
            self.layout = None
        else:
            self.pbar = None

        self.current_tokens = None
        self.mask_positions = None
        self.total_steps = 0
        self.current_step = 0

    def start_visualization(self, initial_tokens: torch.LongTensor, mask_positions: torch.BoolTensor, total_steps: int):
        """
        Start the visualization.

        Args:
            initial_tokens: Initial token IDs (possibly masked)
            mask_positions: Boolean mask indicating which positions are masked
            total_steps: Total number of diffusion steps
        """
        if not self.show_visualization:
            self.pbar = tqdm(total=total_steps, desc="Diffusion")
            return

        self.current_tokens = initial_tokens.clone()
        self.mask_positions = mask_positions
        self.total_steps = total_steps
        self.current_step = 0

        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="text", ratio=1),
            Layout(name="progress", size=3)
        )

        self.progress = Progress(
            TextColumn("[bold blue]Diffusion"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]Masks: {task.fields[masks]}"),
            TimeRemainingColumn(),
        )
        self.progress_task = self.progress.add_task(
            "Generating",
            total=total_steps,
            masks=mask_positions.sum().item()
        )

        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        self.live.start()
        self._update_display()

    def update_step(self, tokens: torch.LongTensor, maskable: Optional[torch.BoolTensor], step: int,
                    entropy: Optional[torch.FloatTensor] = None, confidence: Optional[torch.FloatTensor] = None):
        """
        Update visualization for current step.

        Args:
            tokens: Current token IDs
            maskable: Boolean mask of remaining masked positions
            step: Current step number
            entropy: Optional entropy scores for each position
            confidence: Optional confidence scores for each position
        """
        if not self.show_visualization:
            if self.pbar:
                self.pbar.update(1)
                masks = maskable.sum().item() if maskable is not None else 0
                self.pbar.set_postfix({'masks': masks})
            return

        self.current_tokens = tokens.clone()
        self.mask_positions = maskable
        self.current_step = step

        masks_remaining = maskable.sum().item() if maskable is not None else 0
        self.progress.update(
            self.progress_task,
            advance=1,
            masks=masks_remaining
        )

        self._update_display()

    def _update_display(self):
        """Update the live display."""
        if not self.live:
            return

        header = Text("RND1-Base Generation", style="bold magenta", justify="center")
        self.layout["header"].update(Panel(header, border_style="bright_blue"))

        text_display = self._format_text_with_masks()
        self.layout["text"].update(
            Panel(
                text_display,
                title="[bold]Generated Text",
                subtitle=f"[dim]Step {self.current_step}/{self.total_steps}[/dim]",
                border_style="cyan"
            )
        )

        self.layout["progress"].update(Panel(self.progress))

    def _format_text_with_masks(self) -> Text:
        """
        Format text with colored masks.

        Returns:
            Rich Text object with formatted tokens
        """
        text = Text()

        if self.current_tokens is None:
            return text

        token_ids = self.current_tokens[0] if self.current_tokens.dim() > 1 else self.current_tokens
        mask_flags = self.mask_positions[0] if self.mask_positions is not None and self.mask_positions.dim() > 1 else self.mask_positions

        for i, token_id in enumerate(token_ids):
            if mask_flags is not None and i < len(mask_flags) and mask_flags[i]:
                # Alternate colors for visual effect
                text.append("[MASK]", style="bold red on yellow" if self.current_step % 2 == 0 else "bold yellow on red")
            else:
                try:
                    token_str = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
                    # Skip special tokens in display
                    if token_str not in ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<s>", "</s>"]:
                        # Color based on position
                        text.append(token_str, style="green" if i < len(token_ids) // 2 else "cyan")
                except:
                    continue

        return text

    def stop_visualization(self):
        """Stop the visualization and display final result."""
        if not self.show_visualization:
            if self.pbar:
                self.pbar.close()
                print("\n✨ Generation complete!\n")
            return

        if self.live:
            self.live.stop()

            self.console.print("\n[bold green]✨ Generation complete![/bold green]\n")

            # Display final text
            if self.current_tokens is not None:
                try:
                    token_ids = self.current_tokens[0] if self.current_tokens.dim() > 1 else self.current_tokens
                    final_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

                    self.console.print(Panel(
                        final_text,
                        title="[bold]Final Generated Text",
                        border_style="green",
                        padding=(1, 2)
                    ))
                except:
                    pass


class SimpleProgressBar:
    """
    Simple progress bar fallback when rich is not available.

    Provides basic progress tracking using tqdm when the rich library
    is not installed.
    """

    def __init__(self, total_steps: int):
        """
        Initialize simple progress bar.

        Args:
            total_steps: Total number of steps
        """
        self.pbar = tqdm(total=total_steps, desc="Diffusion")

    def update(self, masks_remaining: int = 0):
        """
        Update progress bar.

        Args:
            masks_remaining: Number of masks still remaining
        """
        self.pbar.update(1)
        self.pbar.set_postfix({'masks': masks_remaining})

    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        print("\n✨ Generation complete!\n")
