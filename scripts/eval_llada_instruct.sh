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
main_port=20002

# ===== Common instruct arguments =====
common_args="--model llada --seed 1234 --device cuda --apply_chat_template"

# =======================
# Instruct Generation Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks gsm8k_cot \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 8 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks bbh \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks minerva_math \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 4 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks humaneval_instruct \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks mbpp_llada_instruct \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 3 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,cfg=0.0"


# =======================
# Instruct Likelihood Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks mmlu_generative \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks mmlu_pro \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks hellaswag_gen \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=3,steps=3,block_length=3,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks arc_challarc_challenge_chatenge \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=5,steps=5,block_length=5,cfg=0.0"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_llada.py \
    --tasks gpqa_n_shot_gen \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,is_check_greedy=False,mc_num=1,max_new_tokens=32,steps=32,block_length=32,cfg=0.0"
