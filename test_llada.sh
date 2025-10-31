export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/datasets/huggingface"
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
export HF_EVALUATE_CACHE="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/.cache/hf_evaluate_rank_${SLURM_PROCID}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true  # For CMMLU dataset

main_port=29515
num_gpu=6
pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16"
common_args="--model llada --seed 1234 --apply_chat_template  --limit 24" #  --limit None
base_path="dllm/eval/eval_llada.py"

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks winogrande \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128" \
#     --device cuda \
#     --num_fewshot 5

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks gsm8k_cot \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
#     --device cuda

# |  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k_cot|      3|flexible-extract|     8|exact_match|↑  |0.7500|±  |0.0778|
# |         |       |strict-match    |     8|exact_match|↑  |0.5625|±  |0.0891|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks mbpp_llada_instruct \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
#     --device cuda \
#     --confirm_run_unsafe_code \
#     --output_path llada_mbpp_llada_instruct_1024.json \
#     --log_samples

# |       Tasks       |Version|   Filter   |n-shot| Metric  |   |Value|   |Stderr|
# |-------------------|------:|------------|-----:|---------|---|----:|---|-----:|
# |mbpp_llada_instruct|      3|extract_code|     3|pass_at_1|↑  |  0.7|±  |0.0205|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks minerva_math \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
#     --device cuda \
#     --output_path llada_minerva_math_16.json \
#     --log_samples \


# "results": {
#     "minerva_math": {
#       "math_verify,none": 0.3392857142857143,
#       "math_verify_stderr,none": 0.029742697335081992,
#       "exact_match,none": 0.026785714285714284,
#       "exact_match_stderr,none": 0.010757401726372752,
#       "alias": "minerva_math"
#     },


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks bbh \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
#     --device cuda \
#     --output_path llada_bbh_16.json \
#     --log_samples \

# |Groups|Version|  Filter  |n-shot|  Metric   |   |Value |   |Stderr|
# |------|------:|----------|------|-----------|---|-----:|---|-----:|
# |bbh   |      3|get-answer|      |exact_match|↑  |0.4907|±  |0.0438|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks humaneval_instruct \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=1024,steps=1024,block_length=32" \
#     --device cuda \
#     --output_path llada_humaneval_16.json \
#     --log_samples \
#     --confirm_run_unsafe_code

# |      Tasks       |Version|  Filter   |n-shot|Metric|   |Value|   |Stderr|
# |------------------|------:|-----------|-----:|------|---|----:|---|-----:|
# |humaneval_instruct|      4|create_test|     0|pass@1|   | 0.75|±  |  0.25|



# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks gpqa_main_n_shot \
#     --batch_size 8 \
#     --num_fewshot 5 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128" \
#     --device cuda


# |     Tasks      |Version|Filter|n-shot| Metric |   |Value|   |Stderr|
# |----------------|------:|------|-----:|--------|---|----:|---|-----:|
# |gpqa_main_n_shot|      2|none  |     5|acc     |↑  | 0.25|±  |0.1118|
# |                |       |none  |     5|acc_norm|↑  | 0.25|±  |0.1118|

# main_port=29516
# num_gpu=4
# pretrained="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16"
# common_args="--model llada --seed 1234 --apply_chat_template  --limit 48" #  --limit None
# base_path="dllm/eval/eval_llada.py"

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks gpqa_n_shot_gen \
#     --batch_size 1 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=128,steps=128,block_length=64" \
#     --device cuda \
#     --output_path llada_gpqa_gen.json \
#     --log_samples

# |     Tasks     |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|
# |---------------|------:|-------------------|-----:|-----------|---|-----:|---|-----:|
# |gpqa_n_shot_gen|      1|gpqa_answer_extract|     5|exact_match|↑  |0.4375|±  |0.1281|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks truthfulqa_mc2 \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=2.0,is_check_greedy=False,mc_num=128" \
#     --device cuda

# |    Tasks     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |--------------|------:|------|-----:|------|---|-----:|---|-----:|
# |truthfulqa_mc2|      3|none  |     0|acc   |↑  |0.5656|±  |0.1253|


# main_port=29516
# num_gpu=4

# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks arc_challenge_chat \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=5,steps=5,block_length=5" \
#     --device cuda

# |      Tasks       |Version|     Filter      |n-shot|  Metric   |   |Value |   |Stderr|
# |------------------|------:|-----------------|-----:|-----------|---|-----:|---|-----:|
# |arc_challenge_chat|      1|remove_whitespace|     0|exact_match|↑  |0.8438|±  |0.0652|


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks hellaswag \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128" \
#     --device cuda

# |  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |---------|------:|------|-----:|--------|---|-----:|---|-----:|
# |hellaswag|      1|none  |     0|acc     |↑  |0.3750|±  |0.1250|
# |         |       |none  |     0|acc_norm|↑  |0.5625|±  |0.1281|


# PYTHONBREAKPOINT=0 \
# srun -p mllm_safety --quotatype=spot --gres=gpu:${num_gpu} --time=01:00:00 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks hellaswag_gen \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=3,steps=3,block_length=3" \
#     --device cuda \
#     --output_path llada_hellaswag_gen.json \
#     --log_samples

# llada (pretrained=/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct,dtype=bfloat16,max_new_tokens=3,steps=3,block_length=3), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: 1                                                                                                                                             
# |    Tasks    |Version|      Filter       |n-shot|  Metric   |   |Value |   |Stderr|                                                                                       
# |-------------|-------|-------------------|-----:|-----------|---|-----:|---|-----:|                                                                                       
# |hellaswag_gen|Yaml   |first_option_filter|     0|exact_match|↑  |0.7665|±  |0.0042|  


# PYTHONBREAKPOINT=0 \
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --main_process_port ${main_port} \
    ${base_path} \
    --tasks winogrande \
    --batch_size 8 \
    --num_fewshot 5 \
    ${common_args} \
    --model_args "pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=128" \
    --device cuda

# |  Tasks   |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
# |----------|------:|------|-----:|------|---|----:|---|-----:|
# |winogrande|      1|none  |     5|acc   |↑  |0.625|±  | 0.061|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks piqa \
#     --batch_size 8 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.5,is_check_greedy=False,mc_num=128" \
#     --device cuda

# |  Groups   |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-----------|------:|------|------|--------|---|-----:|---|-----:|
# |ceval-valid|      2|none  |      |acc     |↑  |0.2712|±  |0.0121|
# |           |       |none  |      |acc_norm|↑  |0.2712|±  |0.0121|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks mmlu \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1" \
#     --device cuda

# |      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |------------------|------:|------|------|------|---|-----:|---|-----:|
# |mmlu              |      2|none  |      |acc   |↑  |0.2840|±  |0.0149|
# | - humanities     |      2|none  |      |acc   |↑  |0.2740|±  |0.0310|
# | - other          |      2|none  |      |acc   |↑  |0.2933|±  |0.0318|
# | - social sciences|      2|none  |      |acc   |↑  |0.2708|±  |0.0325|
# | - stem           |      2|none  |      |acc   |↑  |0.2928|±  |0.0257|


# common_args="--model llada --seed 1234 --apply_chat_template" #  --limit None
# num_gpu=1
# main_port=29516

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks mmlu_generative \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=3,steps=3,block_length=3" \
#     --device cuda

# |      Groups      |Version|   Filter   |n-shot|  Metric   |   |Value |   |Stderr|
# |------------------|------:|------------|------|-----------|---|-----:|---|-----:|
# |mmlu (generative) |      3|get_response|      |exact_match|↑  |0.6667|±  |0.0312|
# | - humanities     |       |get_response|      |exact_match|↑  |0.5962|±  |0.0618|
# | - other          |       |get_response|      |exact_match|↑  |0.6154|±  |0.0684|
# | - social sciences|       |get_response|      |exact_match|↑  |0.8333|±  |0.0538|
# | - stem           |       |get_response|      |exact_match|↑  |0.6447|±  |0.0603|

common_args="--model llada --seed 1234 --apply_chat_template --limit 4" #  --limit None
num_gpu=4

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes ${num_gpu} \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks mmlu_pro \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},max_new_tokens=256,steps=256,block_length=256" \
#     --device cuda \
#     --output_path llada_mmlu_pro_gen.json \
#     --log_samples

# |       Tasks       |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
# |-------------------|------:|--------------|-----:|-----------|---|-----:|---|-----:|
# |mmlu_pro           |    2.0|custom-extract|      |exact_match|↑  |0.3929|±  |0.0565|
# | - biology         |    2.1|custom-extract|     0|exact_match|↑  |0.7500|±  |0.2500|
# | - business        |    2.1|custom-extract|     0|exact_match|↑  |0.5000|±  |0.2887|
# | - chemistry       |    2.1|custom-extract|     0|exact_match|↑  |0.7500|±  |0.2500|
# | - computer_science|    2.1|custom-extract|     0|exact_match|↑  |0.2500|±  |0.2500|
# | - economics       |    2.1|custom-extract|     0|exact_match|↑  |0.5000|±  |0.2887|
# | - engineering     |    2.1|custom-extract|     0|exact_match|↑  |1.0000|±  |0.0000|
# | - health          |    2.1|custom-extract|     0|exact_match|↑  |0.5000|±  |0.2887|
# | - history         |    2.1|custom-extract|     0|exact_match|↑  |0.2500|±  |0.2500|
# | - law             |    2.1|custom-extract|     0|exact_match|↑  |0.0000|±  |0.0000|
# | - math            |    2.1|custom-extract|     0|exact_match|↑  |0.0000|±  |0.0000|
# | - other           |    2.1|custom-extract|     0|exact_match|↑  |0.7500|±  |0.2500|
# | - philosophy      |    2.1|custom-extract|     0|exact_match|↑  |0.0000|±  |0.0000|
# | - physics         |    2.1|custom-extract|     0|exact_match|↑  |0.2500|±  |0.2500|
# | - psychology      |    2.1|custom-extract|     0|exact_match|↑  |0.0000|±  |0.0000|

# | Groups |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
# |--------|------:|--------------|------|-----------|---|-----:|---|-----:|
# |mmlu_pro|      2|custom-extract|      |exact_match|↑  |0.3929|±  |0.0565|
    

# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks cmmlu \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1" \
#     --device cuda

# |Groups|Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |------|------:|------|------|--------|---|-----:|---|-----:|
# |cmmlu |      1|none  |      |acc     |↑  |0.2927|±  |0.0069|
# |      |       |none  |      |acc_norm|↑  |0.2927|±  |0.0069|


# PYTHONBREAKPOINT=0 \
# accelerate launch \
#     --num_processes 4 \
#     --num_machines 1 \
#     --main_process_port ${main_port} \
#     ${base_path} \
#     --tasks ceval-valid \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     ${common_args} \
#     --model_args "pretrained=${pretrained},cfg=0.0,is_check_greedy=False,mc_num=1" \
#     --device cuda

# |  Groups   |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-----------|------:|------|------|--------|---|-----:|---|-----:|
# |ceval-valid|      2|none  |      |acc     |↑  |0.2712|±  |0.0121|
# |           |       |none  |      |acc_norm|↑  |0.2712|±  |0.0121|