# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
RND1 Generation Configuration.

This module defines the generation configuration for RND1 models,
controlling the diffusion-based generation process.
"""

from typing import Optional
from transformers.generation.configuration_utils import GenerationConfig


class RND1GenerationConfig(GenerationConfig):
    """
    Configuration class for RND1 generation parameters.

    This class extends the base GenerationConfig to include parameters
    specific to diffusion-based language generation.

    Args:
        max_length: Maximum sequence length
        num_diffusion_steps: Number of denoising steps in the diffusion process
        mask_token_id: Token ID used for masking during diffusion
        temperature: Temperature for sampling (higher = more random)
        top_k: Optional top-k filtering
        top_p: Optional nucleus (top-p) filtering
        greedy: Whether to use greedy decoding (True) or stochastic sampling (False)
        **kwargs: Additional arguments passed to GenerationConfig
    """

    def __init__(
        self,
        max_length: int = 256,
        num_diffusion_steps: int = 256,
        mask_token_id: int = 151669,
        temperature: float = 0.1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        bos_token_id: int = None,
        eos_token_id: int = None,
        pad_token_id: int = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # Force no caching for RND generation
        # kwargs['use_cache'] = False
        kwargs.pop('use_cache', None)
        super().__init__(
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=not greedy,
            use_cache=False,
            **kwargs,
        )

        # RND-specific parameters
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = mask_token_id
        self.greedy = greedy

    def to_dict(self):
        """Convert configuration to dictionary."""
        output = super().to_dict()
        output["num_diffusion_steps"] = self.num_diffusion_steps
        output["mask_token_id"] = self.mask_token_id
        output["greedy"] = self.greedy
        return output
