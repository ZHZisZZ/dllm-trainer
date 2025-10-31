#!/usr/bin/env bash
#SBATCH --job-name=dllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety
#SBATCH --quotatype=spot
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# ===== Cluster variables =====
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
TRAIN_NODES=("${NODELIST[@]}")

echo "===== System Variables ====="
{
  echo "NUM_NODES=$NUM_NODES"
  echo "GPUS_PER_NODE=$GPUS_PER_NODE"
  echo "WORLD_SIZE=$WORLD_SIZE"
  echo "MASTER_ADDR=$MASTER_ADDR"
  echo "MASTER_PORT=$MASTER_PORT"
} | column -t -s=

echo "Nodes allocated:"
for node in "${TRAIN_NODES[@]}"; do
  echo "  - $node"
done
echo "============================"

# ===== Environment =====
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH=.

# ===== Default options =====
accelerate_config="zero2"
script_path="scripts/examples/llada_sft.py"

# ===== Parse arguments =====
# Stop parsing known options as soon as we hit an unknown one
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerate_config)
      accelerate_config="$2"; shift 2 ;;
    --script_path)
      script_path="$2"; shift 2 ;;
    *)
      FORWARD_ARGS=("$@"); break ;;  # everything else goes to the training script
  esac
done

echo "===== Script Variables ====="
echo "--accelerate_config ${accelerate_config}"
echo "--script_path ${script_path}"
echo "--forwarded script args:"
printf '%s\n' "${FORWARD_ARGS[@]}" | xargs -n 2
echo "============================"

# ===== Launch =====
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --nodelist="${SLURM_JOB_NODELIST}" \
  accelerate launch \
    --config_file "scripts/accelerate_configs/${accelerate_config}.yaml" \
    --num_machines "${NUM_NODES}" \
    --num_processes "${WORLD_SIZE}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${SLURM_PROCID}" \
    --rdzv_backend c10d \
    "${script_path}" "${FORWARD_ARGS[@]}"
