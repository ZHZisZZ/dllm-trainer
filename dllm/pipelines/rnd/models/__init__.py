# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

"""
Radical Numerics Diffusion (RND1) - Diffusion-based Language Model.
"""

from .configuration_rnd import RND1Config
from .modeling_rnd import (
    RND1LM,
    RND1Model,
    RND1PreTrainedModel,
    RND1Attention,
    RND1DecoderLayer,
    RND1SparseMoeBlock,
)
from .generation_config import RND1GenerationConfig
from .generation_utils import RND1GenerationMixin
from .sampling import (
    diffusion_sample,
    apply_top_k_filtering,
    apply_top_p_filtering,
)
from .terminal_visualizer import TerminalVisualizer, SimpleProgressBar

__version__ = "0.1.0"

__all__ = [
    "RND1Config",
    "RND1GenerationConfig",
    "RND1LM",
    "RND1Model",
    "RND1PreTrainedModel",
    "RND1Attention",
    "RND1DecoderLayer",
    "RND1SparseMoeBlock",
    "RND1GenerationMixin",
    "TerminalVisualizer",
    "SimpleProgressBar",
]

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("rnd1", RND1Config)
    AutoModel.register(RND1Config, RND1LM)
    AutoModelForMaskedLM.register(RND1Config, RND1LM)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
