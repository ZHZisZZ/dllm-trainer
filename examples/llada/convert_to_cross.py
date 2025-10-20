import torch
from dataclasses import dataclass
import dllm
from dllm.utils.configs import ModelArguments
from dllm.pipelines.llada.models.modeling_llada_cross import LLaDALlamaBlockWithCross

@dataclass
class ConvertModelArguments(ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    save_path: str = "models/LLaDA-8B-Cross"

def convert_llada_checkpoint(state_dict):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # 기존 레이어들은 그대로 복사
        if any(name in key for name in ['q_proj', 'k_proj', 'v_proj', 'attn_out', 
                                      'ff_proj', 'up_proj', 'ff_out', 
                                      'attn_norm', 'ff_norm']):
            new_state_dict[key] = value
            
        # Cross attention 레이어들은 대응되는 self attention 가중치로 초기화
        if 'q_proj' in key:
            cross_key = key.replace('q_proj', 'cross_q_proj')
            new_state_dict[cross_key] = value.clone()
        if 'k_proj' in key:
            cross_key = key.replace('k_proj', 'cross_k_proj')
            new_state_dict[cross_key] = value.clone()
        if 'v_proj' in key:
            cross_key = key.replace('v_proj', 'cross_v_proj')
            new_state_dict[cross_key] = value.clone()
        if 'attn_out' in key:
            cross_key = key.replace('attn_out', 'cross_attn_out')
            new_state_dict[cross_key] = value.clone()
            
        # Cross attention의 layer norm도 기존 attention norm으로 초기화
        if 'attn_norm' in key:
            cross_key = key.replace('attn_norm', 'cross_attn_norm')
            new_state_dict[cross_key] = value.clone()
    
    return new_state_dict

def convert_model():
    # 인자 파싱
    model_args = ConvertModelArguments()
    
    # 원본 모델 로드
    print(f"Loading original model from {model_args.model_name_or_path}")
    original_model = dllm.utils.get_model(model_args=model_args)
    
    # 새 모델 초기화
    print("Initializing cross attention model")
    cross_model = dllm.utils.get_model(model_args=model_args)
    
    # Cross attention 블록으로 교체
    for i in range(len(cross_model.model.transformer.blocks)):
        print(f"Converting block {i}")
        old_block = cross_model.model.transformer.blocks[i]
        new_block = LLaDALlamaBlockWithCross(
            layer_id=i,
            config=old_block.config,
            cache=old_block._LLaDABlock__cache
        )
        cross_model.model.transformer.blocks[i] = new_block
    
    # 가중치 변환 및 로드
    print("Converting weights")
    new_state_dict = convert_llada_checkpoint(original_model.state_dict())
    cross_model.load_state_dict(new_state_dict, strict=False)
    
    # 모델 저장
    print(f"Saving converted model to {model_args.save_path}")
    cross_model.save_pretrained(model_args.save_path)
    
    print("Conversion complete!")
    return cross_model

if __name__ == "__main__":
    convert_model()
