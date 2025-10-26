from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class GeneratorOutput:
    sequences: torch.Tensor
    histories: list[torch.Tensor] | None = None


@dataclass
class GeneratorConfig:
    return_dict_in_generate: bool = False


@dataclass
class BaseGenerator(ABC):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler()

    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        prompts: list[torch.Tensor, list], 
        config: GeneratorConfig = GeneratorConfig(), 
        **kwargs
    ) -> GeneratorOutput:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor, list], 
        config: GeneratorConfig = GeneratorConfig(), 
        **kwargs
    ) -> GeneratorOutput:
        raise NotImplementedError

