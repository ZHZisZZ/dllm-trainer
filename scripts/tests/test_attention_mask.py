"""
LLaDA / MoE / Dream / RND attention mask invariance tests (compact version)
"""

import gc

import torch
import transformers
import dllm
import pytest

ERROR_THRESHOLD = 1e-3


def _cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Reclaim interprocess memory blocks (useful after large model del)
        try:
            torch.cuda.ipc_collect()
        except Exception:
            # Not all PyTorch builds expose ipc_collect on all platforms
            pass


def _forward_variants(model):
    """
    Run the 5 padding/mask variants and return tensors sliced to the 'real' tokens [1,2,3,4].
    Returns dict: {'A','B','C','D','E'} each [1, 4, H]
    """
    device = model.device

    # A: no padding
    a_ids  = torch.tensor([[1, 2, 3, 4]], device=device)
    a_mask = torch.tensor([[1, 1, 1, 1]], device=device)

    # B: left-pad a 0
    b_ids  = torch.tensor([[0, 1, 2, 3, 4]], device=device)
    b_mask = torch.tensor([[0, 1, 1, 1, 1]], device=device)

    # C: right-pad a 0
    c_ids  = torch.tensor([[1, 2, 3, 4, 0]], device=device)
    c_mask = torch.tensor([[1, 1, 1, 1, 0]], device=device)

    # D: same as A but attention_mask=None
    d_ids  = torch.tensor([[1, 2, 3, 4]], device=device)
    d_mask = None

    # E: same as A but omit attention_mask entirely
    e_ids  = torch.tensor([[1, 2, 3, 4]], device=device)

    with torch.no_grad():
        out_A = model(input_ids=a_ids, attention_mask=a_mask).logits            # [1,4,H]
        out_B = model(input_ids=b_ids, attention_mask=b_mask).logits[:, 1:]     # [1,4,H]
        out_C = model(input_ids=c_ids, attention_mask=c_mask).logits[:, :-1]    # [1,4,H]
        out_D = model(input_ids=d_ids, attention_mask=d_mask).logits            # [1,4,H]
        out_E = model(input_ids=e_ids).logits                                    # [1,4,H]

    return {"A": out_A, "B": out_B, "C": out_C, "D": out_D, "E": out_E}


def _assert_invariance(outs: dict, tag: str):
    ref = outs["A"]
    for k in ("B", "C", "D", "E"):
        assert torch.allclose(ref, outs[k], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD), \
            f"[{tag}] Mismatch A vs {k}"


@pytest.mark.parametrize(
    "repo, attn_impl, human_name",
    [
        ("GSAI-ML/LLaDA-8B-Base",               None,   "LLaDA Base"),
        ("inclusionAI/LLaDA-MoE-7B-A1B-Base",   None,   "LLaDA MoE"),
        ("Dream-org/Dream-v0-Base-7B",          None,   "Dream Base"),
        ("radicalnumerics/RND1-Base-0910",      None,   "RND Base (native)"),
        ("radicalnumerics/RND1-Base-0910",      "sdpa", "RND Base (SDPA)"),
    ],
)
def test_attention_mask_invariance(repo, attn_impl, human_name):
    """
    For each model/backend:
      1) Check padding/mask invariance across A..E on the 'real' tokens.
      2) Print a ✅ message for debug visibility (pytest still enforces assertions).
    """
    model_path = dllm.utils.resolve_with_base_env(repo, "BASE_MODELS_DIR")

    if attn_impl is None:
        model = transformers.AutoModel.from_pretrained(
            model_path, dtype=torch.float32, device_map="auto"
        ).eval()
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_path, attn_implementation=attn_impl
        )
        model = transformers.AutoModel.from_pretrained(
            model_path, config=config, dtype=torch.float32, device_map="auto"
        ).eval()

    outs = _forward_variants(model)
    _assert_invariance(outs, human_name)

    print(f"✅ {human_name} attention mask invariance passed within {ERROR_THRESHOLD}.")
    del model
    gc.collect(); _cuda_cleanup()


def test_rnd_native_vs_sdpa_equivalence():
    """
    Verify RND (native attention) and RND (SDPA) produce equivalent logits on the
    same real tokens across A..E variants.
    """
    repo = "radicalnumerics/RND1-Base-0910"
    model_path = dllm.utils.resolve_with_base_env(repo, "BASE_MODELS_DIR")

    # native
    model_native = transformers.AutoModel.from_pretrained(
        model_path, dtype=torch.float32, device_map="auto"
    ).eval()

    # sdpa
    config_sdpa = transformers.AutoConfig.from_pretrained(
        model_path, attn_implementation="sdpa"
    )
    model_sdpa = transformers.AutoModel.from_pretrained(
        model_path, config=config_sdpa, dtype=torch.float32, device_map="auto"
    ).eval()

    outs_native = _forward_variants(model_native)  # expects helper from your file
    outs_sdpa   = _forward_variants(model_sdpa)

    for k in ("A", "B", "C", "D", "E"):
        assert torch.allclose(
            outs_native[k], outs_sdpa[k], atol=ERROR_THRESHOLD, rtol=ERROR_THRESHOLD
        ), f"[RND cross-backend] native vs SDPA mismatch on {k}"

    print(f"✅ RND native vs SDPA equivalence passed within {ERROR_THRESHOLD}.")
    # Explicitly drop model references
    del model_native
    del model_sdpa
    # Collect Python garbage and release CUDA caches
    gc.collect()
    _cuda_cleanup()


def test_rnd1_attention_mask():
    """
    Verify that the model produces identical logits for the same "real" tokens,
    regardless of left/right padding or whether attention_mask is explicitly given.
    """
    model_name_or_path = dllm.utils.resolve_with_base_env(
        "radicalnumerics/RND1-Base-0910", "BASE_MODELS_DIR"
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
    breakpoint()
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
        f"✅ RND1 attention mask test passed — all variants match within {ERROR_THRESHOLD} tolerance."
    )


test_rnd1_attention_mask()