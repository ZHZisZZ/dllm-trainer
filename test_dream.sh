#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/datasets/huggingface"
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true  # For CMMLU dataset

main_port=29511
pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/Dream-org/Dream-v0-Instruct-7B,dtype=bfloat16"
common_args="--model dream --limit 16 --seed 1234 --apply_chat_template" #  --limit None
base_path="dllm/eval/eval_dream.py"
num_gpu=4


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks gsm8k_cot \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 0 \
#         --output_path dream_gsm8k \
#         --log_samples

# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_cot|      3|flexible-extract|     0|exact_match|↑  |0.7969|±  |0.0507|
# |         |       |strict-match    |     0|exact_match|↑  |0.0000|±  |0.0000|



# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks mbpp_instruct \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 0 \
#         --output_path dream_mbpp_16.json \
#         --log_samples \
#         --confirm_run_unsafe_code

# |    Tasks    |Version|   Filter   |n-shot| Metric  |   |Value |   |Stderr|
# |-------------|------:|------------|-----:|---------|---|-----:|---|-----:|
# |mbpp_instruct|      1|extract_code|     0|pass_at_1|↑  |0.5156|±  | 0.063|


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks minerva_math \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=512,steps=512,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 0

# |   Groups   |Version|Filter|n-shot|  Metric   |   |Value |   |Stderr|
# |------------|------:|------|------|-----------|---|-----:|---|-----:|
# |minerva_math|      1|none  |      |exact_match|↑  |0.0000|±  |0.0000|
# |            |       |none  |      |math_verify|↑  |0.4263|±  |0.0218|


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks humaneval_instruct \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=768,steps=768,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 0 \
#         --confirm_run_unsafe_code

# |      Tasks       |Version|  Filter   |n-shot|Metric|   |Value |   |Stderr|
# |------------------|------:|-----------|-----:|------|---|-----:|---|-----:|
# |humaneval_instruct|      4|create_test|     0|pass@1|   |0.7969|±  |0.0507|


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks ifeval \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=1280,steps=1280,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 0 

# |Tasks |Version|Filter|n-shot|        Metric         |   |Value |   |Stderr|
# |------|------:|------|-----:|-----------------------|---|-----:|---|------|
# |ifeval|      4|none  |     0|inst_level_loose_acc   |↑  |0.7677|±  |   N/A|
# |      |       |none  |     0|inst_level_strict_acc  |↑  |0.7374|±  |   N/A|
# |      |       |none  |     0|prompt_level_loose_acc |↑  |0.6562|±  |0.0598|
# |      |       |none  |     0|prompt_level_strict_acc|↑  |0.6094|±  |0.0615|




PYTHONBREAKPOINT=0 \
srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
    accelerate launch --main_process_port ${main_port} ${base_path} \
        --tasks mmlu_generative \
        --batch_size 1 \
        ${common_args} \
        --model_args "pretrained=${pretrained},max_new_tokens=128,steps=128,temperature=0.1,top_p=0.9,alg=entropy" \
        --device cuda \
        --num_fewshot 4 \
        --confirm_run_unsafe_code

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks mmlu_pro \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},max_new_tokens=128,steps=128,temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 4 \
#         --confirm_run_unsafe_code

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
#     accelerate launch --main_process_port ${main_port} ${base_path} \
#         --tasks gpqa_main_n_shot \
#         --batch_size 1 \
#         ${common_args} \
#         --model_args "pretrained=${pretrained},temperature=0.1,top_p=0.9,alg=entropy" \
#         --device cuda \
#         --num_fewshot 5 \
#         --confirm_run_unsafe_code