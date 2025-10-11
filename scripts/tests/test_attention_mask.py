"""
LLaDA attention mask test
"""

import torch
import transformers
import dllm

ERROR_THRESHOLD = 1e-3


def test_llada_attention_mask():
    """
    Verify that the model produces identical logits for the same "real" tokens,
    regardless of left/right padding or whether attention_mask is explicitly given.
    """
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "GSAI-ML/LLaDA-8B-Base", "BASE_MODELS_DIR"
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, dtype=torch.float32, device_map="auto"
    ).eval()

    # ----- Case A: no padding -----
    input_ids_A = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_A = torch.tensor([[1, 1, 1, 1]], device=model.device)

    # ----- Case B: left-pad with a 0 -----
    input_ids_B = torch.tensor([[0, 1, 2, 3, 4]], device=model.device)
    attn_B = torch.tensor([[0, 1, 1, 1, 1]], device=model.device)

    # ----- Case C: right-pad with a 0 -----
    input_ids_C = torch.tensor([[1, 2, 3, 4, 0]], device=model.device)
    attn_C = torch.tensor([[1, 1, 1, 1, 0]], device=model.device)

    # ----- Case D: same as A but no explicit mask -----
    input_ids_D = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_D = None

    # ----- Case E: same as A but omit attention_mask argument completely -----
    input_ids_E = torch.tensor([[1, 2, 3, 4]], device=model.device)

    # Forward pass
    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A).logits
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B).logits
        out_C = model(input_ids=input_ids_C, attention_mask=attn_C).logits
        out_D = model(input_ids=input_ids_D, attention_mask=attn_D).logits
        out_E = model(input_ids=input_ids_E).logits

    # ----- Compare “real” token positions -----
    assert torch.allclose(
        out_A, out_B[:, 1:], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and left-pad (B) outputs."
    assert torch.allclose(
        out_A, out_C[:, :-1], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and right-pad (C) outputs."
    assert torch.allclose(
        out_A, out_D, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and implicit mask (D) outputs."
    assert torch.allclose(
        out_A, out_E, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and no-mask (E) outputs."

    print(
        f"✅ LLaDA attention mask test passed — all variants match within {ERROR_THRESHOLD} tolerance."
    )


def test_llada_moe_attention_mask():
    """
    Verify that the model produces identical logits for the same "real" tokens,
    regardless of left/right padding or whether attention_mask is explicitly given.
    """
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "inclusionAI/LLaDA-MoE-7B-A1B-Base", "BASE_MODELS_DIR"
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, dtype=torch.float32, device_map="auto"
    ).eval()

    # ----- Case A: no padding -----
    input_ids_A = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_A = torch.tensor([[1, 1, 1, 1]], device=model.device)

    # ----- Case B: left-pad with a 0 -----
    input_ids_B = torch.tensor([[0, 1, 2, 3, 4]], device=model.device)
    attn_B = torch.tensor([[0, 1, 1, 1, 1]], device=model.device)

    # ----- Case C: right-pad with a 0 -----
    input_ids_C = torch.tensor([[1, 2, 3, 4, 0]], device=model.device)
    attn_C = torch.tensor([[1, 1, 1, 1, 0]], device=model.device)

    # ----- Case D: same as A but no explicit mask -----
    input_ids_D = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_D = None

    # ----- Case E: same as A but omit attention_mask argument completely -----
    input_ids_E = torch.tensor([[1, 2, 3, 4]], device=model.device)

    # Forward pass
    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A).logits
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B).logits
        out_C = model(input_ids=input_ids_C, attention_mask=attn_C).logits
        out_D = model(input_ids=input_ids_D, attention_mask=attn_D).logits
        out_E = model(input_ids=input_ids_E).logits

    # ----- Compare “real” token positions -----
    assert torch.allclose(
        out_A, out_B[:, 1:], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and left-pad (B) outputs."
    assert torch.allclose(
        out_A, out_C[:, :-1], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and right-pad (C) outputs."
    assert torch.allclose(
        out_A, out_D, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and implicit mask (D) outputs."
    assert torch.allclose(
        out_A, out_E, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and no-mask (E) outputs."

    print(
        f"✅ LLaDA MoE attention mask test passed — all variants match within {ERROR_THRESHOLD} tolerance."
    )


def test_dream_attention_mask():
    """
    Verify that the model produces identical logits for the same "real" tokens,
    regardless of left/right padding or whether attention_mask is explicitly given.
    """
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "Dream-org/Dream-v0-Base-7B", "BASE_MODELS_DIR"
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, dtype=torch.float32, device_map="auto"
    ).eval()

    # ----- Case A: no padding -----
    input_ids_A = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_A = torch.tensor([[1, 1, 1, 1]], device=model.device)

    # ----- Case B: left-pad with a 0 -----
    input_ids_B = torch.tensor([[0, 1, 2, 3, 4]], device=model.device)
    attn_B = torch.tensor([[0, 1, 1, 1, 1]], device=model.device)

    # ----- Case C: right-pad with a 0 -----
    input_ids_C = torch.tensor([[1, 2, 3, 4, 0]], device=model.device)
    attn_C = torch.tensor([[1, 1, 1, 1, 0]], device=model.device)

    # ----- Case D: same as A but no explicit mask -----
    input_ids_D = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_D = None

    # ----- Case E: same as A but omit attention_mask argument completely -----
    input_ids_E = torch.tensor([[1, 2, 3, 4]], device=model.device)

    # Forward pass
    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A).logits
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B).logits
        out_C = model(input_ids=input_ids_C, attention_mask=attn_C).logits
        out_D = model(input_ids=input_ids_D, attention_mask=attn_D).logits
        out_E = model(input_ids=input_ids_E).logits

    # ----- Compare “real” token positions -----
    assert torch.allclose(
        out_A, out_B[:, 1:], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and left-pad (B) outputs."
    assert torch.allclose(
        out_A, out_C[:, :-1], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between no-pad (A) and right-pad (C) outputs."
    assert torch.allclose(
        out_A, out_D, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and implicit mask (D) outputs."
    assert torch.allclose(
        out_A, out_E, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    ), "Mismatch between explicit mask (A) and no-mask (E) outputs."

    print(
        f"✅ Dream attention mask test passed — all variants match within {ERROR_THRESHOLD} tolerance."
    )
