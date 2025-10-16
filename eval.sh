#!/usr/bin/env bash
#SBATCH --job-name=llada-eval
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --nodes=2
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
MASTER_PORT=$((20000 + 
SLURM_JOB_ID % 10000))

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
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/shared/models/huggingface/"
export HF_DATASETS_CACHE="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface"
export MASTER_ADDR MASTER_PORT WORLD_SIZE

# ===== Optional arguments =====
TASKS="winogrande"
NUM_FEWSHOT=3
BATCH_SIZE=16
SEED=1234
MC_NUM=128
MAX_NEW_TOKENS=512
STEPS=512
BLOCK_LENGTH=512
LIMIT=None
# For Debugging
# SAMPLES="{\"gsm8k\": [$(echo {0..128} | sed 's/ /,/g')]}"

# ===== Launch =====
echo "Launching accelerate on ${NUM_NODES} node(s) (${WORLD_SIZE} GPUs total)..."

if [[ "${NUM_NODES}" -eq 1 ]]; then
  echo "Running single-node setup..."
  accelerate launch \
    --num_processes "${WORLD_SIZE}" \
    --num_machines "${NUM_NODES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${SLURM_PROCID}" \
    dllm/eval/eval_llada.py \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size ${BATCH_SIZE} \
    --model llada_dist \
    --model_args "pretrained=/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_length=${BLOCK_LENGTH}" \
    --tasks ${TASKS} \
    --seed ${SEED} #\
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
      dllm/eval/eval_llada.py \
      --num_fewshot ${NUM_FEWSHOT} \
      --batch_size ${BATCH_SIZE} \
      --model llada_dist \
      --model_args "pretrained=/mnt/lustrenew/mllm_aligned/shared/models/huggingface/GSAI-ML/LLaDA-8B-Base,is_check_greedy=False,mc_num=${MC_NUM},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},block_length=${BLOCK_LENGTH}" \
      --tasks ${TASKS} \
      --seed ${SEED} #\
      # --limit ${LIMIT} 
      # --samples "${SAMPLES}"
fi
