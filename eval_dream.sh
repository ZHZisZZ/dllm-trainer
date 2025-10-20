#!/usr/bin/env bash
#SBATCH --job-name=dream-eval
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue


# ===== Derived variables =====
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

# ===== Node information =====
NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=${NODELIST[0]}
TRAIN_NODES=("${NODELIST[@]}")

echo "============================"
echo "JOB NAME:       ${SLURM_JOB_NAME}"
echo "JOB ID:         ${SLURM_JOB_ID}"
echo "PARTITION:      ${SLURM_JOB_PARTITION}"
echo "NUM_NODES:      ${NUM_NODES}"
echo "GPUS_PER_NODE:  ${GPUS_PER_NODE}"
echo "WORLD_SIZE:     ${WORLD_SIZE}"
echo "MASTER_ADDR:    ${MASTER_ADDR}"
echo "MASTER_PORT:    ${MASTER_PORT}"
echo "============================"

echo "Nodes allocated:"
for node in "${TRAIN_NODES[@]}"; do
  echo "  - $node"
done
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
export MASTER_ADDR MASTER_PORT WORLD_SIZE

# ===== Optional arguments =====
TASKS="gsm8k"   # Separated by commas
NUM_FEWSHOT=8
BATCH_SIZE=1
SEED=1234
MC_NUM=1
MAX_NEW_TOKENS=256
MAX_LENGTH=4096
STEPS=256
TEMPERATURE=0.0
TOP_P=1.0
LIMIT=64
USE_CHAT_TEMPLATE=True
MODEL_NAME="Dream-org/Dream-v0-Instruct-7B"
MODEL_PATH="${BASE_MODELS_DIR}/${MODEL_NAME}" 
# For Debugging
# SAMPLES="{\"gsm8k\": [$(echo {0..128} | sed 's/ /,/g')]}"

# ===== Conditional argument =====
if [[ "${USE_CHAT_TEMPLATE}" == "True" ]]; then
  APPLY_CHAT_TEMPLATE_ARG="--apply_chat_template True --fewshot_as_multiturn"
else
  APPLY_CHAT_TEMPLATE_ARG=""
fi

# ===== Launch =====
echo "Launching accelerate on ${NUM_NODES} node(s) (${WORLD_SIZE} GPUs total)..."
echo "Testing ${MODEL_PATH} on ${TASKS}"
echo "Apply chat template: ${USE_CHAT_TEMPLATE}"
echo "============================"

if [[ "${NUM_NODES}" -eq 1 ]]; then
  echo "Running single-node setup..."
  accelerate launch \
    --num_processes "${WORLD_SIZE}" \
    --num_machines "${NUM_NODES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${SLURM_PROCID}" \
    dllm/eval/eval_dream.py \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size ${BATCH_SIZE} \
    --model dream \
    --model_args "pretrained=${MODEL_PATH},mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},max_length=${MAX_LENGTH},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=true,escape_until=true" \
    --tasks ${TASKS} \
    --seed ${SEED} \
    --log_samples \
    --output_path ./logs/gsm8k_${SLURM_JOB_ID}_samples.json \
    --limit ${LIMIT} \
    ${APPLY_CHAT_TEMPLATE_ARG}
    # --apply_chat_template True \
    # --limit ${LIMIT} 
    # --samples "${SAMPLES}"
else
  echo "Running multi-node setup..."
  srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --nodelist="${SLURM_JOB_NODELIST}" \
    accelerate launch \
      --num_processes "${WORLD_SIZE}" \
      --num_machines "${NUM_NODES}" \
      --main_process_ip "${MASTER_ADDR}" \
      --main_process_port "${MASTER_PORT}" \
      --machine_rank "${SLURM_PROCID}" \
      --rdzv_backend c10d \
      dllm/eval/eval_dream.py \
      --num_fewshot ${NUM_FEWSHOT} \
      --batch_size ${BATCH_SIZE} \
      --model dream \
      --model_args "pretrained=${MODEL_PATH},mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},max_length=${MAX_LENGTH},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},add_bos_token=true,escape_until=true" \
      --tasks ${TASKS} \
      --seed ${SEED} \
      --log_samples \
      --output_path ./logs/gsm8k_${SLURM_JOB_ID}_samples.json \
      --limit ${LIMIT} \
      ${APPLY_CHAT_TEMPLATE_ARG}
      # --apply_chat_template True \
      # --limit ${LIMIT} 
      # --samples "${SAMPLES}"
fi
