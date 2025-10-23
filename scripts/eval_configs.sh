#!/bin/bash
# ============================================================
# Unified configuration file for model evaluation (LLaDA + Dream)
# ============================================================

# Each dataset maps to a tuple of arguments.
# For LLaDA:
# eval_llada_configs["<dataset>"]="num_fewshot|limit|use_chat_template|max_new_tokens|steps|block_length|seed|batch_size|mc_num|cfg"
#
# For Dream:
# eval_dream_configs["<dataset>"]="num_fewshot|limit|use_chat_template|max_new_tokens|max_length|steps|temperature|top_p|seed|batch_size|mc_num"
# ============================================================

declare -A eval_llada_configs
declare -A eval_dream_configs

# ============================================================
# ======== LLaDA configurations ========
# ============================================================

# Base generation
eval_llada_configs["gsm8k"]="8|None|False|1024|1024||1234|1|1|0.0"
eval_llada_configs["bbh"]="4|None|False|1024|1024||1234|1|1|0.0"
eval_llada_configs["minerva_math"]="4|None|False|1024|1024||1234|1|1|0.0"
eval_llada_configs["humaneval"]="0|None|False|1024|1024||1234|1|1|0.0"
eval_llada_configs["mbpp"]="3|None|False|1024|1024||1234|1|1|0.0"

# Base likelihood
eval_llada_configs["gpqa_main_n_shot"]="5|None|False|1024|1024|1024|1234|8|128|0.5"
eval_llada_configs["truthfulqa_mc2"]="0|None|False|1024|1024|1024|1234|8|128|2.0"
eval_llada_configs["arc_challenge"]="0|None|False|1024|1024|1024|1234|8|128|0.5"
eval_llada_configs["hellaswag"]="0|None|False|1024|1024|1024|1234|8|128|0.5"
eval_llada_configs["winogrande"]="5|None|False|1024|1024|1024|1234|8|128|0.0"
eval_llada_configs["piqa"]="0|None|False|1024|1024|1024|1234|8|128|0.5"
eval_llada_configs["mmlu"]="5|None|False|1024|1024|1024|1234|1|1|0.0"
eval_llada_configs["cmmlu"]="5|None|False|1024|1024|1024|1234|1|1|0.0"
eval_llada_configs["ceval-valid"]="5|None|False|1024|1024|1024|1234|1|1|0.0"

# # Instruct generation
# eval_llada_configs["gsm8k"]="8|256|True|1024|1024|32|1234|1|1|0.0"
# eval_llada_configs["bbh"]="4|None|True|1024|1024|32|1234|1|1|0.0"
# eval_llada_configs["minerva_math"]="4|None|True|1024|1024|32|1234|1|1|0.0"
# eval_llada_configs["humaneval"]="0|None|True|1024|1024|32|1234|1|1|0.0"
# eval_llada_configs["mbpp"]="3|None|True|1024|1024|32|1234|1|1|0.0"

# # Instruct likelihood
# eval_llada_configs["gpqa_main_n_shot"]="5|None|True|1024|1024|1024|1234|8|128|0.5"
# eval_llada_configs["truthfulqa_mc2"]="0|None|True|1024|1024|1024|1234|8|128|2.0"
# eval_llada_configs["arc_challenge"]="0|None|True|1024|1024|1024|1234|8|128|0.5"
# eval_llada_configs["hellaswag"]="0|None|True|1024|1024|1024|1234|8|128|0.5"
# eval_llada_configs["winogrande"]="5|None|True|1024|1024|1024|1234|8|128|0.0"
# eval_llada_configs["piqa"]="0|None|True|1024|1024|1024|1234|8|128|0.5"
# eval_llada_configs["mmlu"]="5|None|True|1024|1024|1024|1234|1|1|0.0"
# eval_llada_configs["cmmlu"]="5|None|True|1024|1024|1024|1234|1|1|0.0"
# eval_llada_configs["ceval-valid"]="5|None|True|1024|1024|1024|1234|1|1|0.0"


# ============================================================
# ======== Dream configurations ========
# ============================================================

# Instruct generation
eval_dream_configs["mmlu_generative"]="4|None|True|128|128|128|0.1|0.9|42|1|1"
eval_dream_configs["mmlu_pro"]="4|None|True|128|128|128|0.1|0.9|42|1|1"
eval_dream_configs["gsm8k_cot"]="0|None|True|256|256|256|0.1|0.9|42|1|1"
eval_dream_configs["minerva_math"]="0|None|True|512|512|512|0.1|0.9|42|1|1"
eval_dream_configs["gpqa_main_n_shot"]="5|None|True|128|128|128|0.0|1.0|42|1|1"
eval_dream_configs["humaneval_instruct"]="0|None|True|768|768|768|0.1|0.9|42|1|1"
eval_dream_configs["mbpp_instruct"]="0|None|True|1024|1024|1024|0.1|0.9|42|1|1"
eval_dream_configs["ifeval"]="0|None|True|1280|1280|1280|0.1|0.9|42|1|1"

# Base generation
eval_dream_configs["humaneval"]="0|None|False|512|512|512|0.2|0.95|42|1|1"
eval_dream_configs["gsm8k_cot"]="8|None|False|256|256|256|0.0|0.95|42|1|1"
eval_dream_configs["mbpp"]="3|None|False|512|512|512|0.2|0.95|42|1|1"
eval_dream_configs["minerva_math"]="4|None|False|512|512|512|0.0|0.95|42|1|1"
eval_dream_configs["bbh"]="3|None|False|512|512|512|0.0|0.95|42|1|1"

# Base likelihood
eval_dream_configs["mmlu"]="5|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["arc_easy"]="0|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["arc_challenge"]="0|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["hellaswag"]="0|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["piqa"]="0|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["gpqa_main_n_shot"]="5|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["winogrande"]="5|None|False|512|512|512|0.0|0.95|42|32|1"
eval_dream_configs["race"]="0|None|False|512|512|512|0.0|0.95|42|32|1"