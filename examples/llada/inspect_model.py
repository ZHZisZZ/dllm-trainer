import torch
from dataclasses import dataclass
import dllm
from dllm.utils.configs import ModelArguments

@dataclass
class InspectModelArguments(ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"

def inspect_model():
    # 모델 로드
    model_args = InspectModelArguments()
    model = dllm.utils.get_model(model_args=model_args)
    
    # 전체 모델 구조 출력
    print("=== Full Model Architecture ===")
    print(model)
    
    # 레이어별 상세 정보
    print("\n=== Detailed Layer Information ===")
    
    # Transformer 블록 정보
    transformer = model.model.transformer
    print(f"\nEmbedding dimension: {transformer.wte.embedding_dim}")
    print(f"Number of layers: {len(transformer.blocks)}")
    
    # 첫 번째 블록의 상세 구조 분석
    block0 = transformer.blocks[0]
    print("\nAttention structure in first block:")
    print(f"Q projection: {block0.q_proj}")
    print(f"K projection: {block0.k_proj}")
    print(f"V projection: {block0.v_proj}")
    print(f"Output projection: {block0.attn_out}")
    
    # FFN 구조
    print("\nFFN structure in first block:")
    print(f"First projection: {block0.ff_proj}")  # 4096 -> 12288
    print(f"Up projection: {block0.up_proj}")     # 4096 -> 12288
    print(f"Output projection: {block0.ff_out}")  # 12288 -> 4096
    
    # 각 블록의 파라미터 수 계산
    print("\nParameters per block:")
    for i, block in enumerate(transformer.blocks):
        params = sum(p.numel() for p in block.parameters())
        print(f"Block {i}: {params:,} parameters")

if __name__ == "__main__":
    inspect_model()
