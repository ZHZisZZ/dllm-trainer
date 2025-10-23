#!/usr/bin/env bash
#SBATCH --job-name=model-eval
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue

# ===== Derived variables =====
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=${NODELIST[0]}
TRAIN_NODES=("${NODELIST[@]}")

echo "============================"
echo "JOB NAME: ${SLURM_JOB_NAME}"
echo "JOB ID: ${SLURM_JOB_ID}"
echo "NUM_NODES: ${NUM_NODES}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================"

# ===== Environment =====
export PYTHONBREAKPOINT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/datasets/huggingface"
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
export HF_EVALUATE_CACHE="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/.cache/hf_evaluate_rank_${SLURM_PROCID}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true # For CMMLU
export MASTER_ADDR MASTER_PORT WORLD_SIZE

# ===== Load configs =====
source ./eval_configs.sh

MODEL_CLASS=${1,,}   # "llada" or "dream"
TASK=${2:-"gsm8k"}   # dataset name
MODEL_NAME=${3}      # model path or name (required)

if [[ -z "${MODEL_NAME}" ]]; then
  echo "❌ Missing model name/path argument!"
  echo "Usage: sbatch eval_model.sh <model_class> <task> <model_name_or_path>"
  exit 1
fi

case "${MODEL_CLASS}" in
  llada)
    CONFIG="${eval_llada_configs[$TASK]}"
    if [[ -z "${CONFIG}" ]]; then
      echo "❌ Unknown task '${TASK}' for LLaDA."
      echo "Available tasks: ${!eval_llada_configs[@]}"
      exit 1
    fi

    # ---- Match config order exactly ----
    IFS="|" read -r NUM_FEWSHOT LIMIT USE_CHAT_TEMPLATE MAX_NEW_TOKENS STEPS BLOCK_LENGTH SEED BATCH_SIZE MC_NUM CFG <<< "${CONFIG}"

    MODEL_PATH="${BASE_MODELS_DIR}/${MODEL_NAME}" # direct argument
    MODEL_TYPE="llada_dist"
    SCRIPT_PATH="dllm/eval/eval_llada.py"
    MODEL_ARGS="pretrained=${MODEL_PATH},is_check_greedy=False,mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_length=${BLOCK_LENGTH},cfg=${CFG}"
    ;;

  dream)
    CONFIG="${eval_dream_configs[$TASK]}"
    if [[ -z "${CONFIG}" ]]; then
      echo "❌ Unknown task '${TASK}' for Dream."
      echo "Available tasks: ${!eval_dream_configs[@]}"
      exit 1
    fi

    IFS="|" read -r NUM_FEWSHOT LIMIT USE_CHAT_TEMPLATE MAX_NEW_TOKENS MAX_LENGTH STEPS TEMPERATURE TOP_P SEED BATCH_SIZE MC_NUM <<< "${CONFIG}"

    MODEL_PATH="${BASE_MODELS_DIR}/${MODEL_NAME}"
    MODEL_TYPE="dream"
    SCRIPT_PATH="dllm/eval/eval_dream.py"
    MODEL_ARGS="pretrained=${MODEL_PATH},mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},max_length=${MAX_LENGTH},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=true,escape_until=true"
    ;;

  *)
    echo "❌ Invalid model_class '${MODEL_CLASS}'. Must be 'llada' or 'dream'."
    exit 1
    ;;
esac

# ===== Conditional flags =====
[[ "${USE_CHAT_TEMPLATE}" == "True" ]] && APPLY_CHAT_TEMPLATE_ARG="--apply_chat_template True" || APPLY_CHAT_TEMPLATE_ARG=""
[[ "${LIMIT}" == "None" ]] && LIMIT_ARG="" || LIMIT_ARG="--limit ${LIMIT}"

# ===== Run =====
echo -e "\nLaunching ${MODEL_CLASS} on ${TASK} using ${MODEL_PATH}"
echo "============================"
echo "Few-shot: ${NUM_FEWSHOT}"
echo "Seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "============================"

RUN_CMD="accelerate launch \
  --num_processes ${WORLD_SIZE} \
  --num_machines ${NUM_NODES} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${SLURM_PROCID} \
  ${SCRIPT_PATH} \
  --num_fewshot ${NUM_FEWSHOT} \
  --batch_size ${BATCH_SIZE} \
  --model ${MODEL_TYPE} \
  --model_args \"${MODEL_ARGS}\" \
  --tasks ${TASK} \
  --seed ${SEED} \
  --log_samples \
  --output_path ./logs/${MODEL_CLASS}_${TASK}_${SLURM_JOB_ID}_samples.json \
  --confirm_run_unsafe_code \
  ${LIMIT_ARG} \
  ${APPLY_CHAT_TEMPLATE_ARG}"

if [[ "${NUM_NODES}" -eq 1 ]]; then
  echo "Single-node execution"
  eval ${RUN_CMD}
else
  echo "Multi-node execution"
  srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --nodelist="${SLURM_JOB_NODELIST}" ${RUN_CMD}
fi
