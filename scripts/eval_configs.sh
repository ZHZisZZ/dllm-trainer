#!/bin/bash
# ============================================================
# Unified Evaluation Configuration File
# ============================================================
# This file defines dataset-specific evaluation configurations
# for LLaDA, Dream, and BERT model classes.
# Each configuration follows the specific format documented
# under its respective section below.
# ============================================================


# ------------------------------------------------------------
# Declare associative arrays
# ------------------------------------------------------------
declare -A eval_llada_configs
declare -A eval_dream_configs
declare -A eval_bert_configs


# ============================================================
# ====================  LLaDA CONFIGS  ========================
# ============================================================
# Format:
#   eval_llada_configs["<dataset>"]="num_fewshot|limit|max_new_tokens|steps|block_length|seed|mc_num|cfg"
# ============================================================

# ---------- Base Generation ----------
eval_llada_configs["gsm8k"]="8|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["bbh"]="3|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["minerva_math"]="4|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["humaneval"]="0|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["mbpp"]="3|None|1024|1024|32|1234|1|0.0"

# ---------- Base Likelihood ----------
eval_llada_configs["gpqa_main_n_shot"]="5|None|1024|1024|1024|1234|128|0.5"
eval_llada_configs["truthfulqa_mc2"]="0|None|1024|1024|1024|1234|128|2.0"
eval_llada_configs["arc_challenge"]="0|None|1024|1024|1024|1234|128|0.5"
eval_llada_configs["hellaswag"]="0|None|1024|1024|1024|1234|128|0.5"
eval_llada_configs["winogrande"]="5|None|1024|1024|1024|1234|128|0.0"
eval_llada_configs["piqa"]="0|None|1024|1024|1024|1234|128|0.5"
eval_llada_configs["mmlu"]="5|None|1024|1024|1024|1234|1|0.0"
eval_llada_configs["cmmlu"]="5|None|1024|1024|1024|1234|1|0.0"
eval_llada_configs["ceval-valid"]="5|None|1024|1024|1024|1234|1|0.0"

# ---------- Instruct Generation ----------
eval_llada_configs["gsm8k_cot"]="8|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["bbh"]="3|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["minerva_math"]="4|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["humaneval_instruct"]="0|None|1024|1024|32|1234|1|0.0"
eval_llada_configs["mbpp_llada_instruct"]="3|None|1024|1024|32|1234|1|0.0"

eval_llada_configs["mmlu_generative"]="0|None|3|3|3|1234|1|0.0"
eval_llada_configs["mmlu_pro"]="0|None|256|256|256|1234|1|0.0"
eval_llada_configs["hellaswag_gen"]="0|None|3|3|3|1234|1|0.0"
eval_llada_configs["arc_challarc_challenge_chatenge"]="0|None|5|5|5|1234|1|0.0"
eval_llada_configs["gpqa_n_shot_gen"]="5|None|32|32|32|1234|1|0.0"


# ============================================================
# ====================  DREAM CONFIGS  ========================
# ============================================================
# Format:
#   eval_dream_configs["<dataset>"]="num_fewshot|limit|max_new_tokens|max_length|steps|temperature|top_p|seed|mc_num"
# ============================================================

# ---------- Base Generation ----------
eval_dream_configs["humaneval"]="0|None|512|512|512|0.2|0.95|42|1"
eval_dream_configs["gsm8k_cot"]="8|None|256|256|256|0.0|0.95|42|1"
eval_dream_configs["mbpp"]="3|None|512|512|512|0.2|0.95|42|1"
eval_dream_configs["minerva_math"]="4|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["bbh"]="3|None|512|512|512|0.0|0.95|42|1"

# ---------- Base Likelihood ----------
eval_dream_configs["mmlu"]="5|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["arc_easy"]="0|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["arc_challenge"]="0|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["hellaswag"]="0|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["piqa"]="0|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["gpqa_main_n_shot"]="5|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["winogrande"]="5|None|512|512|512|0.0|0.95|42|1"
eval_dream_configs["race"]="0|None|512|512|512|0.0|0.95|42|1"

# ---------- Instruct Generation ----------
eval_dream_configs["mmlu_generative"]="4|None|128|128|128|0.1|0.9|42|1"
eval_dream_configs["mmlu_pro"]="4|None|128|128|128|0.1|0.9|42|1"
eval_dream_configs["gsm8k_cot"]="0|None|256|256|256|0.1|0.9|42|1"
eval_dream_configs["minerva_math"]="0|None|512|512|512|0.1|0.9|42|1"
eval_dream_configs["gpqa_main_n_shot"]="5|None|128|128|128|0.0|1.0|42|1"
eval_dream_configs["humaneval_instruct"]="0|None|768|768|768|0.1|0.9|42|1"
eval_dream_configs["mbpp_instruct"]="0|None|1024|1024|1024|0.1|0.9|42|1"
eval_dream_configs["ifeval"]="0|None|1280|1280|1280|0.1|0.9|42|1"


# ============================================================
# ====================  BERT CONFIGS  =========================
# ============================================================
# Format:
#   eval_bert_configs["<dataset>"]="num_fewshot|limit|max_new_tokens|steps|block_length|seed|mc_num"
# ============================================================

eval_bert_configs["hellaswag_gen"]="0|100|128|128|128|1234|1"
eval_bert_configs["mmlu_generative"]="0|100|128|128|128|1234|1"
eval_bert_configs["mmlu_pro"]="0|100|256|256|256|1234|1"
eval_bert_configs["arc_challenge_chat"]="0|100|128|128|128|1234|1"
eval_bert_configs["winogrande"]="0|100|128|128|128|1234|1"

# ============================================================
# ======================  END  ===============================
# ============================================================
