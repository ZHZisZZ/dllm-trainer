# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 Generation Utilities.

This module provides generation utilities and mixins for RND1 models,
including the main GenerationMixin class that integrates with HuggingFace.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
from transformers import GenerationMixin as HFGenerationMixin
from transformers.generation import GenerationConfig

from .sampling import diffusion_sample, apply_top_k_filtering, apply_top_p_filtering


class RND1GenerationMixin(HFGenerationMixin):
    """
    Generation mixin for RND1 models.

    This mixin provides generation methods compatible with HuggingFace's
    generation API while using RND1's diffusion-based sampling internally.
    """

    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        # RND1-specific parameters
        prefix_ids: Optional[torch.LongTensor] = None,
        suffix_ids: Optional[torch.LongTensor] = None,
        infill_length: Optional[int] = None,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,  # Accept all kwargs to be compatible with pipelines
    ) -> Union[torch.LongTensor, Dict[str, Any]]:
        """
        Generate text using RND1's diffusion-based sampling.

        Follows HuggingFace's standard generate API, using diffusion sampling
        internally. Supports both standard generation and infilling.

        Args:
            inputs: Input token IDs to use as prefix (standard HF parameter)
            generation_config: Generation configuration object
            prefix_ids: Alternative to inputs for infilling tasks
            suffix_ids: Optional suffix for infilling tasks
            infill_length: Length of infill region (for infilling)
            return_dict_in_generate: Whether to return GenerateDecoderOnlyOutput
            **kwargs: Additional arguments (accepted for compatibility)

        Returns:
            Generated token IDs or GenerateDecoderOnlyOutput
        """
        if generation_config is not None:
            gen_config = generation_config
            model_kwargs = kwargs.copy()
        else:
            # Only prepare config from kwargs if no config was provided
            gen_config, model_kwargs = self._prepare_generation_config(None, **kwargs)

        device = next(self.parameters()).device

        if inputs is not None:
            prefix_ids = inputs.to(device)
        elif prefix_ids is not None:
            prefix_ids = prefix_ids.to(device)
        else:
            prefix_ids = None

        if suffix_ids is not None:
            suffix_ids = suffix_ids.to(device)

        eos_token_id = gen_config.eos_token_id or getattr(self.config, "eos_token_id", 151645)
        eos_token_id = None if eos_token_id == -1 else eos_token_id
        pad_token_id = gen_config.pad_token_id or getattr(self.config, "pad_token_id", None)
        bos_token_id = gen_config.bos_token_id or getattr(self.config, "bos_token_id", None)
        mask_token_id = getattr(gen_config, "mask_token_id", getattr(self.config, "mask_token_id", 151669))

        if infill_length is not None and prefix_ids is not None:
            # Infilling mode: use specified infill_length
            prefix_len = prefix_ids.shape[1] if prefix_ids is not None else 0
            suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
            seq_len = prefix_len + infill_length + suffix_len
        else:
            # Standard generation mode
            if prefix_ids is not None:
                prefix_len = prefix_ids.shape[1]
                if gen_config.max_new_tokens is not None:
                    seq_len = prefix_len + gen_config.max_new_tokens
                else:
                    seq_len = gen_config.max_length or self.config.max_position_embeddings
            else:
                seq_len = gen_config.max_length or self.config.max_position_embeddings

        num_diffusion_steps = getattr(gen_config, "num_diffusion_steps",
                                     getattr(self.config, "num_diffusion_steps", 256))

        temperature = float(getattr(gen_config, "temperature", 1.0))
        top_k = getattr(gen_config, "top_k", None)
        top_p = getattr(gen_config, "top_p", None)

        greedy = getattr(gen_config, "greedy",
                        not bool(gen_config.do_sample) if hasattr(gen_config, "do_sample") else True)


        with torch.inference_mode():
            sequences = diffusion_sample(
                model=self,
                seq_len=seq_len,
                num_steps=num_diffusion_steps,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                prefix_ids=prefix_ids,
                suffix_ids=suffix_ids,
                infill_length=infill_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                device=device,
                visualizer=model_kwargs.get("visualizer", None),  # Optional visualizer from kwargs
            )

        if return_dict_in_generate or getattr(gen_config, "return_dict_in_generate", False):
            from transformers.generation.utils import GenerateDecoderOnlyOutput
            return GenerateDecoderOnlyOutput(sequences=sequences)

        return sequences

    def generate_with_visualization(
        self,
        tokenizer,
        inputs: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        suffix_ids: Optional[torch.LongTensor] = None,
        infill_length: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate with live visualization (for demos).

        This method requires a tokenizer to display the generation process.
        For production use, prefer `generate()`.

        Args:
            tokenizer: Tokenizer for decoding tokens to text
            inputs: Input token IDs to use as prefix
            generation_config: Generation configuration object
            suffix_ids: Optional suffix token IDs
            infill_length: Length of infill region
            **kwargs: Additional arguments for backward compatibility

        Returns:
            Generated token IDs as LongTensor
        """
        from .terminal_visualizer import TerminalVisualizer
        visualizer = TerminalVisualizer(tokenizer, show_visualization=True)

        return self.generate(
            inputs=inputs,
            generation_config=generation_config,
            suffix_ids=suffix_ids,
            infill_length=infill_length,
            visualizer=visualizer,
            return_dict_in_generate=False,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation (required by HuggingFace).

        For RND1, we don't use the standard autoregressive generation,
        so this just returns the input_ids.
        """
        return {"input_ids": input_ids}
