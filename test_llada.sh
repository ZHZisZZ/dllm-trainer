#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/datasets/huggingface"
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true  # For CMMLU dataset

num_gpu=8



# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:000 --exclude=SH-IDCA1404-10-140-54-15 \
#     accelerate launch --main_process_port 29510 dllm/eval/eval_llada.py \
#         --tasks gsm8k \
#         --batch_size 1 \
#         --model llada_dist \
#         --model_args pretrained='/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct',is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16 \
#         --apply_chat_template True \
#         --limit 16 \
#         --seed 1234

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=03:00:000 \
#     accelerate launch --num_processes ${num_gpu} -m lm_eval \
#     --model hf \
#     --model_args pretrained=/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen3-0.6B \
#     --tasks mmlu_college_mathematics,mmlu_high_school_chemistry\
#     --device cuda:0 \
#     --batch_size 5

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:000 \
#     accelerate launch --main_process_port 29510 dllm/eval/eval_llada.py \
#         --tasks minerva_math \
#         --batch_size 1 \
#         --num_fewshot 5 \
#         --model llada_dist \
#         --model_args pretrained='/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct',is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=1024,dtype=bfloat16 \
#         --apply_chat_template True \
#         --limit 4 \
#         --seed 1234



# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:000 \
#     accelerate launch --main_process_port 29510 dllm/eval/eval_llada.py \
#         --tasks bbh \
#         --batch_size 1 \
#         --model llada_dist \
#         --model_args pretrained='/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Base',is_check_greedy=False,mc_num=1,max_new_tokens=256,steps=256,block_length=256,dtype=bfloat16 \
#         --limit 4 \
#         --seed 1234


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:000 \
#     accelerate launch --main_process_port 29510 dllm/eval/eval_dream.py \
#         --model dream \
#         --model_args "pretrained=/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,temperature=0.0,top_p=1.0,add_bos_token=true,escape_until=true,mc_num=1" \
#         --tasks gsm8k \
#         --batch_size 1 \
#         --num_fewshot 8 \
#         --log_samples \
#         --output_path gsm8k_test_sample.json \
#         --seed 1234 \
#         --samples '{"gsm8k": [24]}' \
#         --apply_chat_template \
#         --fewshot_as_multiturn

        # --apply_chat_template False \
        # --limit 32 \
# mmlu_college_mathematics
#  --samples '{"gsm8k": [2]}' \



# MODEL_PATH="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Dream-org/Dream-v0-Instruct-7B"

# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 python gsm8k_eval.py \
#     --model_name ${MODEL_PATH} \
#     --k 8




#!/usr/bin/env bash
# ============================================================
# Evaluate LLaDA-8B-Instruct baseline on multiple reasoning benchmarks
# using distributed accelerate + srun on mllm_safety partition
# ============================================================

num_gpu=8
main_port=29510
pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Base"
common_args="--model llada_dist --limit 64 --seed 1234"  # --apply_chat_template True 
base_path="dllm/eval/eval_llada.py"

# ========== GPQA ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks gpqa_main_n_shot \
#         --num_fewshot 5 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"

# # ========== TruthfulQA ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks truthfulqa_mc2 \
#         --num_fewshot 0 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=2.0,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== ARC-Challenge ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks arc_challenge \
#         --num_fewshot 0 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== HellaSwag ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks hellaswag \
#         --num_fewshot 0 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== Winogrande ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks winogrande \
#         --num_fewshot 5 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== PIQA ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks piqa \
#         --num_fewshot 0 \
#         --batch_size 8 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== MMLU ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks mmlu \
#         --num_fewshot 5 \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== CMMLU ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks cmmlu \
#         --num_fewshot 5 \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16

# # ========== CEval-valid ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks ceval-valid \
#         --num_fewshot 5 \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16





# ============================================================
# Generative Tasks (LLaDA-8B-Instruct)
# ============================================================

main_port=29513
pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16"
common_args="--model llada_dist --limit 16 --seed 1234 --apply_chat_template"
base_path="dllm/eval/eval_llada.py"
num_gpu=2

# # ========== BBH ==========
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks bbh \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"

# # ========== GSM8K ==========
# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks gsm8k_cot \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"

# # ========== Minerva Math ==========
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks minerva_math \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"

# # ========== HumanEval ==========
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks humaneval \
#         --batch_size 1 \
#         ${common_args} \
#         --confirm_run_unsafe_code \
#         --model_args "pretrained=${pretrained},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"

# # ========== MBPP ==========
PYTHONBREAKPOINT=0 \
srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
    accelerate launch --main_process_port ${main_port} ${base_path} \
        --tasks mbpp_instruct \
        --batch_size 1 \
        ${common_args} \
        --confirm_run_unsafe_code \
        --model_args "pretrained=${pretrained},is_check_greedy=False,mc_num=1,max_new_tokens=1024,steps=1024,block_length=32,dtype=bfloat16"
