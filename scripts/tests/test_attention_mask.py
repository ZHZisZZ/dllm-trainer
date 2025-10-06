"""
LLaDA attention mask test
"""

import torch
import transformers
import dllm

ERROR_THRESHOLD = 1e-3


def test_llada_attention_mask():
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "GSAI-ML/LLaDA-8B-Base", "BASE_MODELS_DIR"
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    ).eval()

    # Case A: no padding
    input_ids_A = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_A = torch.tensor([[1, 1, 1, 1]], device=model.device)

    # Case B: left pad with a 0
    input_ids_B = torch.tensor([[0, 1, 2, 3, 4]], device=model.device)
    attn_B = torch.tensor([[0, 1, 1, 1, 1]], device=model.device)

    # Case C: both sides pad with 0
    input_ids_C = torch.tensor([[1, 2, 3, 4, 0]], device=model.device)
    attn_C = torch.tensor([[1, 1, 1, 1, 0]], device=model.device)

    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A).logits
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B).logits
        out_C = model(input_ids=input_ids_C, attention_mask=attn_C).logits

    # Compare the logits for the “real” positions [1,2,3,4]
    assert torch.allclose(
        out_A, out_B[:, 1:], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )
    assert torch.allclose(
        out_A, out_C[:, :-1], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )
    expected_slice = torch.tensor(
        [
            [11.0661, 9.1255, 11.7845],
            [7.3237, 4.7777, 14.1884],
            [9.7602, 6.6605, 7.9520],
            [9.1387, 5.9180, 8.3237],
        ],
        device=out_A.device,
    )
    assert torch.allclose(
        out_A[:, :, :3], expected_slice, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )


def test_llada_moe_attention_mask():
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "inclusionAI/LLaDA-MoE-7B-A1B-Base", "BASE_MODELS_DIR"
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    ).eval()

    # Case A: no padding
    input_ids_A = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attn_A = torch.tensor([[1, 1, 1, 1]], device=model.device)

    # Case B: left pad with a 0
    input_ids_B = torch.tensor([[0, 1, 2, 3, 4]], device=model.device)
    attn_B = torch.tensor([[0, 1, 1, 1, 1]], device=model.device)

    # Case C: both sides pad with 0
    input_ids_C = torch.tensor([[1, 2, 3, 4, 0]], device=model.device)
    attn_C = torch.tensor([[1, 1, 1, 1, 0]], device=model.device)

    with torch.no_grad():
        out_A = model(input_ids=input_ids_A, attention_mask=attn_A).logits
        out_B = model(input_ids=input_ids_B, attention_mask=attn_B).logits
        out_C = model(input_ids=input_ids_C, attention_mask=attn_C).logits

    # Compare the logits for the “real” positions [1,2,3,4]
    assert torch.allclose(
        out_A, out_B[:, 1:], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )
    assert torch.allclose(
        out_A, out_C[:, :-1], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )
    expected_slice = torch.tensor(
        [
            [3.3327, 12.4816, 12.4992],
            [1.5626, 8.7846, 13.9236],
            [2.5232, 9.0491, 13.1045],
            [3.4953, 9.1782, 13.7100],
        ],
        device=out_A.device,
    )
    assert torch.allclose(
        out_A[:, :, :3], expected_slice, atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
    )
