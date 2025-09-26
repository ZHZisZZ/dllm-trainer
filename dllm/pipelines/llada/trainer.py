from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from dllm.utils.schedulers import BaseScheduler, LinearScheduler


class LLaDATrainer(transformers.Trainer):

    def __init__(
        self,
        *args,
        scheduler: BaseScheduler = LinearScheduler(),
        **kawrgs,
    ):
        self.scheduler = scheduler
        return super().__init__(*args, **kawrgs)

    def compute_loss(
        self,
        model: Union[transformers.PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs = False,
        **kwargs,
    ):
        # Reference: https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
        input_ids, labels = inputs["input_ids"], inputs["labels"]

        b, l = inputs["input_ids"].shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = 1 - self.scheduler(t).unsqueeze(1).repeat(1, l)  # b, 1
        loss_weight = - self.scheduler.loss_weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        masked_indices = masked_indices & (labels != -100)
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).repeat(1, l)  # b, l

        noised_input_ids = torch.where(masked_indices, self.processing_class.mask_token_id, input_ids)
        outputs = model(input_ids=noised_input_ids) 
        logits = outputs.logits

        if not masked_indices.any():
            # return a zero loss that retains graph/device/dtype
            return logits.sum() * 0.0

        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none')
        token_loss *= loss_weight[masked_indices]
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / input_ids.shape[0]

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    pass
