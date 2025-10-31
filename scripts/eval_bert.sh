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
main_port=20005

# ===== Common arguments =====
common_args="--model bert --seed 1234 --device cuda --apply_chat_template"

# =======================
# BERT Instruct (Chat) Tasks
# =======================

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_bert.py \
    --tasks hellaswag_gen \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=ModernBERT-base/checkpoint-final,is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_bert.py \
    --tasks mmlu_generative \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=ModernBERT-base/checkpoint-final,is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_bert.py \
    --tasks mmlu_pro \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=ModernBERT-base/checkpoint-final,is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_bert.py \
    --tasks arc_challenge_chat \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=ModernBERT-base/checkpoint-final,is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"

accelerate launch \
    --num_processes ${num_gpu} \
    --num_machines 1 \
    --main_process_port ${main_port} \
    dllm/eval/eval_bert.py \
    --tasks winogrande \
    --batch_size 1 \
    ${common_args} \
    --num_fewshot 0 \
    --model_args "pretrained=ModernBERT-base/checkpoint-final,is_check_greedy=False,mc_num=1,max_new_tokens=128,steps=128,block_length=128"
