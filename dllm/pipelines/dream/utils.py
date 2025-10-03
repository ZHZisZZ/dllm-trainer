from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import transformers

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

@dataclass
class DreamSFTCollator(transformers.DataCollatorForSeq2Seq):
    resp_cutoff_ratio: float = 0.0
    
    def __call__(self, features, return_tensors=None):
        orig_seq_lens = [len(f["input_ids"]) for f in features]
        batch = super().__call__(features, return_tensors=return_tensors)
        if self.resp_cutoff_ratio > 0 and np.random.rand() < self.resp_cutoff_ratio:
            # response lengths are stored per-sample in features
            resp_lens = torch.tensor([len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=torch.long)
            min_resp_len = resp_lens.min().item()
            if min_resp_len > 1:
                # Sample a random cutoff length between [1, min_resp_len)
                cutoff_len = int(np.random.randint(1, min_resp_len))
                orig_seq_len = max(orig_seq_lens)
                new_seq_len = orig_seq_len - cutoff_len

                # Truncate tensors
                for key in ["input_ids", "labels", "attention_mask"]:
                    if key in batch:
                        batch[key] = batch[key][:, :new_seq_len].contiguous()
        if "prompt_len" in batch:
            batch.pop("prompt_len", None)
        return batch
