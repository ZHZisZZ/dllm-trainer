"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

import transformers

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: list[torch.Tensor],
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler(),
    steps: int = 128,
    max_new_tokens: int = 256,
    max_length: int = 1024,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    cfg_keep_tokens: list = None,
    remasking: str = "random",
    return_dict_in_generate: bool = False,
    stochastic_transfer: bool = False,
) -> torch.Tensor | dict:
    """
    Blockwise diffusion-style masked decoding for *generation from prompts*.

    High-level:
      1) Right after each prompt, append `max_new_tokens` mask tokens.
      2) Split the appended tail into blocks of size `block_length`.
      3) For each block, iteratively "reveal" a scheduler-decided number of masked
         positions per step, choosing which positions to commit via a confidence
         score (MaskGIT-style) or randomly.

    Args:
        model (PreTrainedModel):
            Mask predictor that returns logits of shape [B, T, V].
        tokenizer (PreTrainedTokenizer):
            Must provide `eos_token_id` and `mask_token_id`
            (e.g., via `tokenizer.convert_tokens_to_ids("<|mdm_mask|>")`).
        prompts (list[torch.Tensor]):
            List of token-id tensors, each shaped [L_i]. For each sample, the
            L_i prompt tokens are copied into a canvas and then `max_new_tokens`
            mask tokens are appended (these are the targets to be filled).
        steps (int, default=128):
            Global sampling-step budget; redistributed evenly across
            `num_blocks = ceil(max_new_tokens / block_length)` as `ceil(steps / num_blocks)`.
            Some blocks may skip steps if the scheduler yields zero transfers.
        max_new_tokens (int, default=256):
            Number of tokens to generate per sample (i.e., number of appended masks).
            If set, the final sequence width is computed as
            `max_length = max(prompt_lens) + max_new_tokens`.
        max_length (int, default=512):
            Total sequence length of the canvas. Used if `max_new_tokens` is not
            provided; in that case `max_new_tokens = max_length - max(prompt_lens)`.
        block_length (int, default=128):
            Size of the active remasking window over the appended mask tail.
            Must satisfy `1 <= block_length <= max_new_tokens`.
        temperature (float, default=0.0):
            Gumbel-Max temperature applied to logits before argmax; `0` disables noise.
        cfg_scale (float, default=0.0):
            Classifier-free guidance scale. If `> 0`, we run an unconditional
            forward pass by masking original prompt tokens and combine as:
            `un_logits + (cfg_scale + 1) * (logits - un_logits)`.
        remasking (str, default='random'):
            Strategy for choosing which masked positions to commit at each step:
              - 'low_confidence': use p(x0) from softmax(logits) as confidence;
                higher p is revealed first (MaskGIT-style).
              - 'random': select positions uniformly at random.
        return_dict_in_generate (bool, default=False):
            If `True`, return a dict with extra metadata; otherwise return only
            the final token tensor.
        stochastic_transfer (bool, default=False):
            If `True`, the number of tokens revealed per step is sampled from a
            Binomial according to the reverse transfer probability; otherwise
            use the deterministic expectation.

    Returns:
        torch.Tensor | dict:
            If `return_dict_in_generate=False`: tensor `[B, T]` where `T = max_length`
            (with `max_length` resolved from `max_new_tokens` if provided).
            If `True`: dict with keys:
              - "sequences": `[B, T]` final token ids
              - "effective_steps_per_block": list[int], actual unmasking steps per block
    """
    # TODO: Implement blockwise attention mask.
    #       When processing block i, the model must not attend to block i+1.
    assert 1 <= block_length <= max_new_tokens
    assert 1 <= steps

    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
    prompt_lens = [p.shape[0] for p in prompts]

    if max_new_tokens:
        max_length = max_new_tokens + max(prompt_lens)
    else:
        max_new_tokens = max_length - max(prompt_lens)

    B = len(prompts)
    T = max_length

    # ----- Initialize canvas with EOS, copy prompts, and append mask tail -----
    x = torch.full((B, T), eos_id, dtype=torch.long, device=model.device)
    for i, p in enumerate(prompts):
        x[i, : prompt_lens[i]] = p  # keep original prompt tokens
        x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
            mask_id  # append `max_new_tokens` masks to be generated
        )

    # Tokens that were *given* at the start (non-mask, non-EOS).
    # These will be masked in the unconditional forward pass for CFG.
    # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
    unmasked_index = (x != mask_id) & (x != eos_id)
    if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
        keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=model.device))
        unmasked_index = unmasked_index & ~keep_mask

    # ----- Block scheduling over the appended mask tail -----
    num_blocks = math.ceil(max_new_tokens / block_length)
    steps = math.ceil(steps / num_blocks)  # per-block step budget
    effective_steps_per_block: list[int] = []

    for b in range(num_blocks):
        # Build a per-sample mask *within this block* (aligned to each prompt's tail)
        block_mask_index = torch.zeros(
            (B, block_length), dtype=torch.bool, device=x.device
        )

        for j in range(B):
            start = prompt_lens[j] + b * block_length
            end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                width = end - start
                block_mask_index[j, :width] = (
                    x[j, start:end] == mask_id
                )  # which positions in this block are still masked

        # Decide how many tokens to reveal per step in this block
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index,
            steps=steps,
            scheduler=scheduler,
            stochastic=stochastic_transfer,
        )

        # Some steps may be skipped if there are no transfers
        effective_steps = num_transfer_tokens.size(1)
        effective_steps_per_block.append(effective_steps)

        # ----- Iterative reveal inside the current block -----
        for i in range(effective_steps):
            mask_index = x == mask_id  # current global mask map

            # Optional CFG: second forward where original prompt tokens are masked out
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[unmasked_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Argmax decoding with optional Gumbel-Max noise for exploration
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T] predicted token ids

            # Per-position confidence used to pick which masks to commit this step
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # [B, T] confidence of predicted token
            elif remasking == "random":
                x0_p = torch.rand(
                    (x0.shape[0], x0.shape[1]), device=x0.device
                )  # random scores
            else:
                raise NotImplementedError(remasking)

            # Restrict selection window to the *current block's* tail region
            for j in range(B):
                x0_p[j, prompt_lens[j] + (b + 1) * block_length :] = -np.inf

            # Only allow updates at currently masked positions; keep others fixed
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p, -np.inf
            )  # consider masked positions only

            # Pick exactly `num_transfer_tokens[j, i]` highest-confidence positions per sample
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            # Commit chosen predictions into the canvas
            x[transfer_index] = x0[transfer_index]

    # ----- Output format -----
    if not return_dict_in_generate:
        return x
    else:
        return {
            "effective_steps_per_block": effective_steps_per_block,
            "sequences": x,
        }


@torch.no_grad()
def generate_single_cfg(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: list[torch.Tensor],
    q_len: list[int],
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler(),
    steps: int = 128,
    max_new_tokens: int = 256,
    max_length: int = 1024,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    cfg_keep_tokens: list = None,
    remasking: str = "random",
    return_dict_in_generate: bool = False,
    stochastic_transfer: bool = False,
) -> torch.Tensor | dict:
    """
    Blockwise diffusion-style masked decoding with CFG using question-only vs question+answer.

    CFG is computed as:
      - Conditional: question + llm_answer → generate max_new_tokens
      - Unconditional: question only → generate max_new_tokens
      - Final logits = un_logits + (cfg_scale + 1) * (cond_logits - un_logits)

    Args:
        prompts (list[torch.Tensor]):
            Full prompts (question + llm_answer) for each sample.
        q_len (list[int]):
            Length of question-only part for each sample (to extract unconditional prompt).
        Other args similar to generate().
    """
    assert 1 <= block_length <= max_new_tokens
    assert 1 <= steps
    assert cfg_scale > 0.0, "This function requires cfg_scale > 0"

    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    B = len(prompts)
    
    # ----- Setup conditional (question + answer) canvas -----
    cond_prompt_lens = [p.shape[0] for p in prompts]
    T_cond = max(cond_prompt_lens) + max_new_tokens
    
    x_cond = torch.full((B, T_cond), eos_id, dtype=torch.long, device=model.device)
    for i, p in enumerate(prompts):
        x_cond[i, :cond_prompt_lens[i]] = p
        x_cond[i, cond_prompt_lens[i]:cond_prompt_lens[i] + max_new_tokens] = mask_id
    
    # ----- Setup unconditional (question only) canvas -----
    uncond_prompt_lens = q_len  # question lengths
    T_uncond = max(uncond_prompt_lens) + max_new_tokens
    
    x_uncond = torch.full((B, T_uncond), eos_id, dtype=torch.long, device=model.device)
    for i in range(B):
        x_uncond[i, :uncond_prompt_lens[i]] = prompts[i][:uncond_prompt_lens[i]]
        x_uncond[i, uncond_prompt_lens[i]:uncond_prompt_lens[i] + max_new_tokens] = mask_id

    # ----- Block scheduling -----
    num_blocks = math.ceil(max_new_tokens / block_length)
    steps_per_block = math.ceil(steps / num_blocks)
    effective_steps_per_block: list[int] = []

    for b in range(num_blocks):
        # Build block mask for conditional (they share the same generation region)
        block_mask_index = torch.zeros((B, block_length), dtype=torch.bool, device=x_cond.device)
        
        for j in range(B):
            start = cond_prompt_lens[j] + b * block_length
            end = min(start + block_length, cond_prompt_lens[j] + max_new_tokens, T_cond)
            if start < end:
                width = end - start
                block_mask_index[j, :width] = (x_cond[j, start:end] == mask_id)

        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index,
            steps=steps_per_block,
            scheduler=scheduler,
            stochastic=stochastic_transfer,
        )

        effective_steps = num_transfer_tokens.size(1)
        effective_steps_per_block.append(effective_steps)

        # ----- Iterative reveal -----
        for step_i in range(effective_steps):
            # Forward pass for both conditional and unconditional
            cond_logits = model(x_cond).logits  # [B, T_cond, V]
            uncond_logits = model(x_uncond).logits  # [B, T_uncond, V]
            
            # Align logits: extract generation region (both have same max_new_tokens)
            # We need to apply CFG on the generation region logits
            cond_gen_logits = torch.stack([
                cond_logits[j, cond_prompt_lens[j]:cond_prompt_lens[j] + max_new_tokens]
                for j in range(B)
            ], dim=0)  # [B, max_new_tokens, V]
            
            uncond_gen_logits = torch.stack([
                uncond_logits[j, uncond_prompt_lens[j]:uncond_prompt_lens[j] + max_new_tokens]
                for j in range(B)
            ], dim=0)  # [B, max_new_tokens, V]
            
            # Apply CFG on generation region
            # cfg_gen_logits = uncond_gen_logits + (cfg_scale + 1) * (cond_gen_logits - uncond_gen_logits)
            cfg_gen_logits = cfg_scale * cond_gen_logits + (1 - cfg_scale) * uncond_gen_logits
            
            # Reconstruct full logits for conditional sequence (we only update x_cond)
            # For positions outside generation region, use original cond_logits
            final_logits = cond_logits.clone()
            for j in range(B):
                final_logits[j, cond_prompt_lens[j]:cond_prompt_lens[j] + max_new_tokens] = cfg_gen_logits[j]

            # Argmax decoding with optional Gumbel noise on CFG-combined logits
            logits_with_noise = add_gumbel_noise(final_logits, temperature=temperature)
            x0_cond = torch.argmax(logits_with_noise, dim=-1)  # [B, T_cond]
            
            # Also get predictions from unconditional logits (for updating x_uncond independently)
            uncond_logits_with_noise = add_gumbel_noise(uncond_logits, temperature=temperature)
            x0_uncond = torch.argmax(uncond_logits_with_noise, dim=-1)  # [B, T_uncond]

            # Compute confidence for remasking (based on CFG logits)
            if remasking == "low_confidence":
                p = F.softmax(final_logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0_cond.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0_cond, dtype=torch.float32)
            else:
                raise NotImplementedError(remasking)

            # Restrict to current block
            mask_index_cond = x_cond == mask_id
            for j in range(B):
                x0_p[j, cond_prompt_lens[j] + (b + 1) * block_length:] = -np.inf

            # Only update masked positions
            x0_cond = torch.where(mask_index_cond, x0_cond, x_cond)
            confidence = torch.where(mask_index_cond, x0_p, -np.inf)

            # Select top-k positions to commit
            transfer_index_cond = torch.zeros_like(x0_cond, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer_tokens[j, step_i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index_cond[j, select_index] = True

            # Commit CFG predictions to conditional canvas
            x_cond[transfer_index_cond] = x0_cond[transfer_index_cond]
            
            # Update unconditional canvas with CFG predictions (same tokens as conditional)
            for j in range(B):
                # Find which positions in generation region were updated in conditional
                cond_gen_start = cond_prompt_lens[j]
                uncond_gen_start = uncond_prompt_lens[j]
                
                gen_transfer_mask = transfer_index_cond[j, cond_gen_start:cond_gen_start + max_new_tokens]
                
                if gen_transfer_mask.any():
                    # Get the corresponding positions in unconditional canvas
                    uncond_gen_positions = torch.arange(
                        uncond_gen_start, 
                        uncond_gen_start + max_new_tokens, 
                        device=x_uncond.device
                    )[gen_transfer_mask]
                    
                    # Update with CFG predictions (same as conditional)
                    cfg_predicted_tokens = x0_cond[j, cond_gen_start:cond_gen_start + max_new_tokens][gen_transfer_mask]
                    x_uncond[j, uncond_gen_positions] = cfg_predicted_tokens

    if not return_dict_in_generate:
        return x_cond
    else:
        return {
            "effective_steps_per_block": effective_steps_per_block,
            "sequences": x_cond,
        }


@torch.no_grad()
def infilling(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    inputs_with_blanks: list[torch.Tensor],
    scheduler: BaseAlphaScheduler = LinearAlphaScheduler(),
    steps: int = 128,
    block_length: int | None = None,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    cfg_keep_tokens: list = None,
    remasking: str = "random",
    return_dict_in_generate: bool = False,
    stochastic_transfer: bool = False,
) -> torch.Tensor | dict:
    """
    Fill in-place the <|mdm_mask|> tokens contained in `inputs_with_blanks`.
    The whole (padded) sequence is split into block windows of length
    `block_length`; within each window we progressively "unmask" positions
    according to the scheduler and chosen remasking strategy.

    Notes:
    - Right padding uses EOS.
    - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
      unconditional branch, identical to `generate`.
    - Only masked positions are ever updated; non-mask tokens are left intact.
    """
    # TODO: attention mask to avoid looking at the padding eos
    #       (short sequences in the batch).
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # ----- Build canvas: right-pad with EOS to the max length in the batch -----
    B = len(inputs_with_blanks)
    seq_lens = [t.shape[0] for t in inputs_with_blanks]
    T = max(seq_lens)

    # Default to a single block spanning the whole sequence
    if block_length is None:
        block_length = T

    assert 1 <= block_length
    assert 1 <= steps

    x = torch.full((B, T), eos_id, dtype=torch.long, device=device)
    for i, t in enumerate(inputs_with_blanks):
        x[i, : seq_lens[i]] = t

    # Tokens that were *given* at the start (non-mask, non-EOS).
    # These will be masked in the unconditional forward pass for CFG.
    # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
    unmasked_index = (x != mask_id) & (x != eos_id)
    if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
        keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=device))
        unmasked_index = unmasked_index & ~keep_mask

    # ----- Blockwise schedule over the *entire* (padded) sequence -----
    num_blocks = math.ceil(T / block_length)
    steps_per_block = math.ceil(steps / num_blocks)
    effective_steps_per_block: list[int] = []

    for b in range(num_blocks):
        start = b * block_length
        stop = min(start + block_length, T)

        # Per-sample view of which positions in this block are masks
        block_mask_index = torch.zeros(
            (B, block_length), dtype=torch.bool, device=device
        )
        widths = []
        for j in range(B):
            # Width limited by sample's true length and sequence end
            width = max(0, min(seq_lens[j], stop) - start)
            widths.append(width)
            if width > 0:
                block_mask_index[j, :width] = x[j, start : start + width] == mask_id

        # Decide how many tokens to reveal at each step in this block
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index,
            steps=steps_per_block,
            scheduler=scheduler,
            stochastic=stochastic_transfer,
        )

        # Some blocks may have no masks => effective_steps == 0
        effective_steps = num_transfer_tokens.size(1)
        effective_steps_per_block.append(effective_steps)

        for s in range(effective_steps):
            mask_index_full = x == mask_id

            # ----- Forward pass (+ optional CFG) -----
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[unmasked_index] = mask_id  # mask out originally known tokens
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Greedy with optional Gumbel-Max noise
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

            # Confidence used for choosing which masks to commit this step
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                    -1
                )  # [B, T]
            elif remasking == "random":
                x0_p = torch.rand((B, T), device=device)
            else:
                raise NotImplementedError(remasking)

            # Restrict selection to the *current* block only
            for j in range(B):
                end_j = start + widths[j]
                # Outside current block => impossible to select
                x0_p[j, :start] = -np.inf
                x0_p[j, end_j:] = -np.inf

            # Only consider currently-masked positions as candidates
            x0 = torch.where(mask_index_full, x0, x)
            confidence = torch.where(mask_index_full, x0_p, -np.inf)

            # Pick exactly num_transfer_tokens[j, s] positions per sample
            transfer_index = torch.zeros_like(x, dtype=torch.bool)
            for j in range(B):
                k = int(num_transfer_tokens[j, s].item())
                if k > 0:
                    _, select_idx = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_idx] = True

            # Commit selected predictions into the canvas
            x[transfer_index] = x0[transfer_index]

    if not return_dict_in_generate:
        return x
    else:
        return {
            "effective_steps_per_block": effective_steps_per_block,
            "sequences": x,
        }
