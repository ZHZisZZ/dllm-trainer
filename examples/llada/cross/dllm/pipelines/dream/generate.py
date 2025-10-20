"""
reference: https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.distributions as dists

import transformers
from transformers.utils import ModelOutput

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.pipelines.dream.utils import top_p_logits, top_k_logits


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor | None = None
    history: tuple[torch.FloatTensor] | None = None


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@torch.no_grad()
def generate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: list[torch.Tensor],
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler(),
    generation_tokens_hook_func=lambda step, x, logits: x,
    generation_logits_hook_func=lambda step, x, logits: logits,
    output_history: bool = False,
    return_dict_in_generate: bool = False,
    max_new_tokens: int = 256,
    max_length: int = 512,
    mask_token_id: int | None = None,
    steps: int = 50,
    eps: float = 1e-3,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    stochastic_transfer: bool = False,
) -> DreamModelOutput | torch.LongTensor:
    """
    Diffusion-style masked decoding for *generation from prompts*.
    (docstring unchanged)
    """
    # --- Initialization ---
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    prompt_lens = [p.shape[0] for p in prompts]
    if max_new_tokens:
        max_length = max_new_tokens + max(prompt_lens)
    else:
        max_new_tokens = max_length - max(prompt_lens)

    B = len(prompts)
    T = max_length
    x = torch.full((B, T), eos_token_id, dtype=torch.long, device=model.device)

    seq_length = []
    for i, p in enumerate(prompts):
        total_len = prompt_lens[i] + max_new_tokens
        seq_length.append(total_len)
        start = T - total_len
        x[i, start : start + prompt_lens[i]] = p
        x[i, start + prompt_lens[i] : T] = mask_token_id

    attention_mask = torch.zeros((B, T), dtype=torch.float32, device=model.device)
    for j, L in enumerate(seq_length):
        if L > 0:
            attention_mask[j, -L:] = 1.0  # Mandate to be left-padding

    histories = [] if (return_dict_in_generate and output_history) else None

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        pos_id = attention_mask.long().cumsum(-1) - 1
        pos_id.masked_fill_(attention_mask == 0, 1)
    else:
        pos_id = None

    mask_index = x == mask_token_id
    num_transfer_tokens_list = get_num_transfer_tokens(
        mask_index=mask_index,
        steps=steps,
        scheduler=scheduler,
        stochastic=stochastic_transfer,
    )
    effective_steps = num_transfer_tokens_list.size(1)

    # --- Iterative refinement ---
    x = generation_tokens_hook_func(None, x, None)
    for i in range(effective_steps):
        mask_index = x == mask_token_id

        logits = model(x, attention_mask, pos_id).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = generation_logits_hook_func(i, x, logits)

        mask_logits = logits[mask_index]

        if alg == "maskgit_plus":
            confidence, x0 = sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
        elif alg == "topk_margin":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=True,
            )
        elif alg == "entropy":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                neg_entropy=True,
            )
        else:
            raise RuntimeError(f"Unknown alg: {alg}")

        full_confidence = torch.full_like(
            x, -torch.inf, device=model.device, dtype=logits.dtype
        )
        full_confidence[mask_index] = confidence

        for j in range(full_confidence.shape[0]):
            number_transfer_tokens = num_transfer_tokens_list[j, i]
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(
                        full_confidence[j], number_transfer_tokens
                    )
                else:
                    fc = full_confidence[j] / alg_temp
                    fc = F.softmax(fc, dim=-1)
                    transfer_index = torch.multinomial(
                        fc, num_samples=number_transfer_tokens
                    )

                x_ = torch.full_like(x, mask_token_id, device=model.device)
                x_[mask_index] = x0.clone()
                x[j, transfer_index] = x_[j, transfer_index]

        x = generation_tokens_hook_func(i, x, logits)
        if histories is not None:
            histories.append(x.clone())

    if return_dict_in_generate:
        return DreamModelOutput(sequences=x, history=histories)
    else:
        return x


@torch.no_grad()
def infilling(
    model,
    tokenizer,
    inputs_with_blanks: list[torch.Tensor],
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler(),
    generation_tokens_hook_func=lambda step, x, logits: x,
    generation_logits_hook_func=lambda step, x, logits: logits,
    output_history: bool = False,
    return_dict_in_generate: bool = False,
    steps: int = 50,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    stochastic_transfer: bool = False,
) -> DreamModelOutput | torch.LongTensor:
    """
    Fill in-place the tokenizer's `<mask>` tokens contained in `inputs_with_blanks`.
    The whole (right-aligned) canvas is denoised iteratively: at each step, a scheduler
    decides how many masked positions to commit, and a confidence rule (`alg`)
    selects *which* positions to reveal (MaskGIT-style). Non-mask tokens are never changed.

    High-level:
      1) Build a right-aligned canvas per sample (left side padded with EOS).
      2) Compute a per-sample transfer schedule via `scheduler` and `steps`.
      3) At each step: forward pass → AR-shift logits → score masked positions
         via `alg` → choose indices to commit (top-k or soft sampling) → write tokens.

    Notes:
      - Right padding uses EOS (serves as pad here).
      - Only `[MASK]` positions are updated; original tokens remain intact.
      - Logits are AR-shifted to preserve next-token prediction alignment.

    Args:
        model:
            Mask predictor; returns logits of shape [B, T, V] when called as
            `model(x, attention_mask, pos_id)`.
        tokenizer:
            Must provide `mask_token_id` and `eos_token_id`.
        inputs_with_blanks:
            List of 1D LongTensors (token ids). Each may contain `<mask>` tokens
            to be filled; other tokens are treated as fixed context.
        scheduler (BaseAlphaScheduler):
            Controls how many masks to commit per step (deterministic or stochastic).
        generation_tokens_hook_func / generation_logits_hook_func:
            Optional hooks to intercept tokens/logits at each step.
        output_history (bool):
            If True, save intermediate canvases at each step.
        return_dict_in_generate (bool):
            If True, return `DreamModelOutput(sequences, history)`, else only `[B, T]`.
        steps (int):
            Total reverse-diffusion steps (quality–speed trade-off).
        alg (str):
            Confidence rule to rank masked positions:
              - "maskgit_plus": softmax probs
              - "topk_margin": top1 - top2 margin
              - "entropy": negative entropy
        alg_temp (float):
            Temperature for *confidence-based index sampling* (when > 0, soft selection).
        temperature / top_p / top_k:
            Token sampling hyperparameters within `sample_tokens`.
        stochastic_transfer (bool):
            If True, sample the number of transfers per step (Binomial); else use expectation.

    Returns:
        DreamModelOutput | torch.LongTensor:
            If `return_dict_in_generate=True`, returns
              - sequences: `[B, T]` final tokens
              - history:   optional list of intermediate canvases
            Otherwise returns only `[B, T]`.
    """
    device = model.device
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id

    B = len(inputs_with_blanks)
    seq_lens = [t.shape[0] for t in inputs_with_blanks]
    T = max(seq_lens)

    # Build right-aligned canvas; left side padded with EOS (used as pad)
    x = torch.full((B, T), eos_token_id, dtype=torch.long, device=device)
    for i, t in enumerate(inputs_with_blanks):
        L = seq_lens[i]
        x[i, -L:] = t

    # Build 1D attention mask (valid tokens on the right)
    attention_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    for j, L in enumerate(seq_lens):
        if L > 0:
            attention_mask[j, -L:] = True

    # Expand to pairwise attention if left padding is present
    if torch.any(attention_mask == 0.0):
        pos_id = attention_mask.long().cumsum(-1) - 1
        pos_id.masked_fill_(attention_mask == 0, 1)
    else:
        pos_id = None
        attention_mask = "full"
    histories = [] if (return_dict_in_generate and output_history) else None

    # Precompute per-sample transfer schedule (how many to commit per step)
    mask_index = x == mask_token_id
    num_transfer_tokens_list = get_num_transfer_tokens(
        mask_index=mask_index,
        steps=steps,
        scheduler=scheduler,
        stochastic=stochastic_transfer,
    )
    effective_steps = num_transfer_tokens_list.size(1)

    # Optional initial token hook
    x = generation_tokens_hook_func(None, x, None)

    for i in range(effective_steps):
        mask_index = x == mask_token_id

        # Forward pass, then AR-shift to predict token at position i+1
        logits = model(x, attention_mask, pos_id).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = generation_logits_hook_func(i, x, logits)

        # Logits restricted to current `[MASK]` positions
        mask_logits = logits[mask_index]

        # Confidence scoring for masked positions
        if alg == "maskgit_plus":
            confidence, x0 = sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
        elif alg == "topk_margin":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                margin_confidence=True,
            )
        elif alg == "entropy":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                neg_entropy=True,
            )
        else:
            raise RuntimeError(f"Unknown alg: {alg}")

        # Scatter per-position confidence back to full canvas
        full_confidence = torch.full_like(
            x, -torch.inf, device=device, dtype=logits.dtype
        )
        full_confidence[mask_index] = confidence

        # Commit the scheduled number of tokens per sample
        for j in range(B):
            number_transfer_tokens = num_transfer_tokens_list[j, i]
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(
                        full_confidence[j], number_transfer_tokens
                    )
                else:
                    fc = full_confidence[j] / alg_temp
                    fc = F.softmax(fc, dim=-1)
                    transfer_index = torch.multinomial(
                        fc, num_samples=number_transfer_tokens
                    )

                # Candidate tokens at masked positions only
                x_ = torch.full_like(x, mask_token_id, device=device)
                x_[mask_index] = x0.clone()
                x[j, transfer_index] = x_[j, transfer_index]

        # Optional token hook + history logging
        x = generation_tokens_hook_func(i, x, logits)
        if histories is not None:
            histories.append(x.clone())

    if return_dict_in_generate:
        return DreamModelOutput(sequences=x, history=histories)
    else:
        return x
