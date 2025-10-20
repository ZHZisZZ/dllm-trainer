#!/bin/bash
# ============================================================
# Unified configuration file for llada-eval
# Each dataset maps to a tuple of arguments:
# dataset_configs["<dataset>"]="num_fewshot|limit|use_chat_template|max_new_tokens|steps|block_length|model_name|seed|batch_size|mc_num"
# ============================================================

declare -A dataset_configs

# ===== Example configurations =====
dataset_configs["gsm8k"]="8|64|False|1024|1024|1024|GSAI-ML/LLaDA-8B-Base|1234|1|1"
dataset_configs["minerva_math"]="4|16|False|1024|1024|1024|GSAI-ML/LLaDA-8B-Base|1234|1|1"
dataset_configs["humaneval"]="0|4|False|1024|1024|1024|GSAI-ML/LLaDA-8B-Base|1234|1|1"
dataset_configs["mbpp"]="3|64|False|1024|1024|1024|GSAI-ML/LLaDA-8B-Base|1234|1|1"
# dataset_configs["mmlu"]="5|None|False|512|512|64|GSAI-ML/LLaDA-8B-Base|1234|1|1"

# You can add or override datasets like:
# dataset_configs["newtask"]="num_fewshot|limit|use_chat_template|max_new_tokens|steps|block_length|model_name|seed|batch_size|mc_num"
