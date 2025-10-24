#!/usr/bin/env bash
# ============================================================
# Evaluate selected LLaDA checkpoints on all datasets directly via sbatch
# Each checkpoint gets its own log subdirectory under ./logs/<checkpoint>/
# Also evaluate the baseline Instruct model under ./logs/instruct/
# ============================================================

BASE_DIR="/mnt/lustrenew/mllm_aligned/shared/models/tmp/LLaDA-8B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5-gbl"
INSTRUCT_MODEL="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
EVAL_SCRIPT="scripts/eval.slurm.sh"
CONFIG_SCRIPT="scripts/eval_configs.sh"
LOG_ROOT="./logs"

# ===== Load dataset configs =====
source "${CONFIG_SCRIPT}"

# ===== Explicitly chosen checkpoints =====
CHECKPOINTS=(
  "checkpoint-5073"
  "checkpoint-4806"
  "checkpoint-4539"
  "checkpoint-4272"
  "checkpoint-4005"
  "checkpoint-3738"
)

echo "============================"
echo "Evaluating on checkpoints:"
printf '  - %s\n' "${CHECKPOINTS[@]}"
echo "============================"
echo "Datasets:"
printf '  - %s\n' "${!eval_llada_configs[@]}"
echo "============================"

# ===== Function to submit evaluations =====
submit_evals() {
  local model_name="$1"
  local model_path="$2"
  local log_dir="$3"

  mkdir -p "${log_dir}"
  echo ">>> Submitting jobs for model: ${model_name}"
  echo "    Logs will be saved under: ${log_dir}"

  for dataset in "${!eval_llada_configs[@]}"; do
    OUT_LOG="${log_dir}/${dataset}-%j.out"
    ERR_LOG="${log_dir}/${dataset}-%j.err"

    echo "  → sbatch -o ${OUT_LOG} -e ${ERR_LOG} ${EVAL_SCRIPT} llada ${dataset} ${model_path}"
    sbatch -o "${OUT_LOG}" -e "${ERR_LOG}" "${EVAL_SCRIPT}" llada "${dataset}" "${model_path}"

    sleep 1  # prevent scheduler flooding
  done

  echo ""
}

# ===== Evaluate fine-tuned checkpoints =====
# for ckpt in "${CHECKPOINTS[@]}"; do
#   CKPT_PATH="${BASE_DIR}/${ckpt}"
#   CKPT_LOG_DIR="${LOG_ROOT}/${ckpt}"
#   submit_evals "${ckpt}" "${CKPT_PATH}" "${CKPT_LOG_DIR}"
# done

# ===== Evaluate baseline Instruct model =====
INSTRUCT_LOG_DIR="${LOG_ROOT}/instruct"
submit_evals "instruct" "${INSTRUCT_MODEL}" "${INSTRUCT_LOG_DIR}"

echo "✅ All sbatch jobs submitted successfully."
