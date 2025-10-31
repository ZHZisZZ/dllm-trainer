#!/usr/bin/env bash
# =====  Environmental Variables =====
export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True 
export MASTER_ADDR MASTER_PORT WORLD_SIZE

# ===== Basic Settings =====
num_gpu=4
main_port=20003

# ===== Common base arguments =====
common_args="--model dream --seed 42 --device cuda"

# =======================
# Dream Base Generation Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks humaneval \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks gsm8k_cot \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 8 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=256,max_length=256,steps=256,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks mbpp \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.2,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks minerva_math \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 4 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks bbh \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"


# =======================
# Dream Base Likelihood Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks mmlu \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks arc_easy \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks arc_challenge \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks hellaswag \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks piqa \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks gpqa_main_n_shot \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks winogrande \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_dream.py \
    --tasks race \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,mc_num=1,max_new_tokens=512,max_length=512,steps=512,temperature=0.0,top_p=0.95,add_bos_token=true,escape_until=true"
