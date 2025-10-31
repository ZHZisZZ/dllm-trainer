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
main_port=20001

# ===== Common base arguments =====
common_args="--model llada --seed 1234 --device cuda"

# =======================
# Base Generation Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks gsm8k \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 8 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks bbh \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks minerva_math \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 4 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks humaneval \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks mbpp \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"


# =======================
# Base Likelihood Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks gpqa_main_n_shot \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks truthfulqa_mc2 \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=2.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks arc_challenge \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks hellaswag \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks winogrande \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks piqa \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.5"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks mmlu \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks cmmlu \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks ceval-valid \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,cfg=0.0"
