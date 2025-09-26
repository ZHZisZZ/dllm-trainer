#!/bin/bash
#SBATCH --job-name=dllm
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mllm_safety # Note: adjust this for your cluster
#SBATCH --quotatype=spot        # Note: adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# args for multi node training
NUM_NODES=${SLURM_NNODES}
GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
TRAIN_NODES=("${NODELIST[@]}")

echo "===== System Variables ====="
{
    echo "NUM_NODES=$NUM_NODES"
    echo "GPUS_PER_NODE=$GPUS_PER_NODE"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "NODELIST=$NODELIST"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "TRAIN_NODES=$TRAIN_NODES"
} | column -t -s=

echo "Nodes allocated:"
for node in "${TRAIN_NODES[@]}"; do
    echo "  - $node"
done
echo "============================"

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE=disabled
export PYTHONPATH=.

# default args for dllm
accelerate_config="deepspeed_zero3"
script_path="scripts/examples/llada_sft.py"
script_args=""

# parse kwargs from command line
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --accelerate_config) accelerate_config="$2"; shift ;;
        --script_path) script_path="$2"; shift ;;
        --script_args) script_args="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "===== Script Variables ====="
echo "--accelerate_config ${accelerate_config}"
echo "--script_path ${script_path}"
echo "--script_args"
echo "$script_args" | xargs -n 2
echo "============================"

if [[ "$NUM_NODES" -eq 1 ]]; then
    echo "Running single-node setup..."
    accelerate launch \
        --config_file scripts/accelerate_configs/${accelerate_config}.yaml \
        --num_machines ${NUM_NODES} \
        --num_processes ${WORLD_SIZE} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${SLURM_PROCID} \
        ${script_path} ${script_args}
else
    echo "Running multi-node setup..."
    srun --nodes=${NUM_NODES} --ntasks=${NUM_NODES} --nodelist=${TRAIN_NODES} accelerate launch \
        --config_file scripts/accelerate_configs/${accelerate_config}.yaml \
        --num_machines ${NUM_NODES} \
        --num_processes ${WORLD_SIZE} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${SLURM_PROCID} \
        --rdzv_backend c10d \
        ${script_path} ${script_args}
fi
