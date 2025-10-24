import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute how many tokens to transition at each step of the reverse process.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def attn_generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking='low_confidence',
    mask_id=126336,
    pad_token_id=None,  # <-- NEW: pass tokenizer.pad_token_id here (or leave None)
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (B, L).
        steps: Sampling steps (<= gen_length).
        gen_length: Generated answer length.
        block_length: Semi-autoregressive block length (<= gen_length).
        temperature: Sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: 'low_confidence' or 'random'.
        mask_id: Token id of [MASK] (default: 126336).
        pad_token_id: Token id used for padding. Only pads are masked out in attention.
    '''
    # Create initial sequence filled with [MASK]
    B = prompt.shape[0]
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # === Build attention mask (bi-directional; mask ONLY pads) ===
    # If pad_token_id is None -> there are no pad tokens; use all ones.
    if pad_token_id is None:
        attention_mask = torch.ones_like(x, dtype=torch.long, device=x.device)
    else:
        attention_mask = (x != pad_token_id).long()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.0:
                # Build unconditional variant by masking the prompt region (NOT padding!)
                un_x = x.clone()
                un_x[prompt_index] = mask_id

                # Attention masks for both branches:
                # - They should be identical except for PADs (which never change here).
                # - [MASK] positions are NOT PAD, so they remain attendable (mask=1).
                if pad_token_id is None:
                    attn_mask_x = attention_mask
                    attn_mask_un = attention_mask
                else:
                    attn_mask_x = attention_mask
                    # un_x changed tokens to [MASK] in the prompt region, but PADs remain identical:
                    attn_mask_un = (un_x != pad_token_id).long()

                x_ = torch.cat([x, un_x], dim=0)
                attn_mask_ = torch.cat([attn_mask_x, attn_mask_un], dim=0)

                logits = model(x_, attention_mask=attn_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L_total)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Prevent selecting tokens beyond current semi-AR block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # Only replace where current token is [MASK]
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # attention_mask does NOT need to be recomputed:
            # - [MASK] is not PAD
            # - PAD positions never change
            # If you ever feed batches with real PADs changing over time, recompute here.

    return x



@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    import torch
    import transformers
    import dllm

    device = 'cuda'

    model_name_or_path = dllm.utils.resolve_with_base_env(
    "GSAI-ML/LLaDA-8B-Instruct", "BASE_MODELS_DIR"
)
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # Use tokenizer.pad_token_id if it exists; otherwise pass None (no pads to mask)
    pad_token_id = tokenizer.eos_token_id

    out = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking='low_confidence',
        # pad_token_id=pad_token_id,  # <-- NEW
    )

    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    breakpoint()

if __name__ == '__main__':
    main()
