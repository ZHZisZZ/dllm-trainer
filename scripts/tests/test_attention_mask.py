"""
LLaDA attention mask test
"""
import torch
import transformers
import dllm

model_name_or_path = dllm.utils.resolve_with_base_env(
    "GSAI-ML/LLaDA-8B-Instruct", "BASE_MODELS_DIR"
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
print(torch.allclose(out_A, out_B[:, 1:], atol=1e-5, rtol=1e-5))
print(torch.allclose(out_A, out_C[:, :-1], atol=1e-5, rtol=1e-5))


"""
Llama attention mask test
"""
import torch
import transformers
import dllm

# Load model (AutoModelForCausalLM instead of AutoModel so logits are available)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "/mnt/lustrenew/mllm_safety-shared/pipelines/huggingface/meta-llama/Llama-3.2-3B-Instruct", torch_dtype=torch.float32, device_map="auto"
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
print(torch.allclose(out_A, out_B[:, 1:], atol=1e-5, rtol=1e-5))
print(torch.allclose(out_A, out_C[:, :-1], atol=1e-5, rtol=1e-5))

