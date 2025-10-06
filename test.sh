#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export BASE_DATASETS_DIR="/mnt/lustrenew/mllm_aligned/datasets/huggingface"
export BASE_MODELS_DIR="/mnt/lustrenew/mllm_aligned/models/huggingface"

# sbatch  --nodes=2 --exclude=SH-IDCA1404-10-140-54-9,SH-IDCA1404-10-140-54-5 scripts/train.slurm.sh \
#     --accelerate_config "deepspeed_zero2" \
#     --script_path "examples/editflow/adapt_llada.py"


# # Define variables clearly
# learning_rate="5e-5"
# batch_size="3"
# sampler="sample_x0_mixture" # sample_x0_masks, sample_x0_empty, sample_x0_noisy, sample_x0_mixture

# # Build script arguments separately for readability
# script_args=(
#   --learning_rate "${learning_rate}"
#   --output_dir "/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/EditFlow-LLaDA-8B-Instruct/tulu-3-sft-mixture[train:10000,test:1000]-${learning_rate}-${sampler}"
#   --per_device_train_batch_size "${batch_size}"
#   --per_device_eval_batch_size "${batch_size}"
#   --dataset_args "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"
#   --model_name_or_path "/mnt/lustrenew/mllm_aligned/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
#   --x0_sampler "${sampler}"
# )
# script_args_str="${script_args[*]}"
# # echo $script_args_str

# # Submit job with array expansion (quotes preserve spacing)
# sbatch --nodes=2 scripts/train.slurm.sh \
#     --accelerate_config "deepspeed_zero2" \
#     --script_path "examples/editflow/adapt_llada.py" \
#     --script_args "${script_args_str}"



#!/usr/bin/env bash

# 1️⃣ Parse choice argument (default = 'mixture')
choice="${1:-mixture}"
mask_length="${2:-0}"

# 2️⃣ Base model directory (no trailing slash)
base_dir="/mnt/lustrenew/mllm_safety-shared/tmp/fanyuyu/models/EditFlow-LLaDA-8B-Instruct"

# 3️⃣ Build model path depending on choice
if [[ "$choice" == "empty" ]]; then
    model_path="${base_dir}/tulu-3-sft-mixture-train10000-test1000-5e-5-sample_x0_empty/checkpoint-final"
elif [[ "$choice" == "masks" ]]; then
    model_path="${base_dir}/tulu-3-sft-mixture-train10000-test1000-5e-5-sample_x0_masks/checkpoint-final"
elif [[ "$choice" == "mixture" ]]; then
    model_path="${base_dir}/tulu-3-sft-mixture-train10000-test1000-5e-5-sample_x0_mixture/checkpoint-final"
elif [[ "$choice" == "noisy" ]]; then
    model_path="${base_dir}/tulu-3-sft-mixture-train10000-test1000-5e-5-sample_x0_noisy/checkpoint-final"
else
    echo "❌ Invalid choice: '$choice'. Must be one of: empty | masks | mixture | noisy"
    exit 1
fi

# 4️⃣ Info
echo "🚀 Using model: $model_path"
echo "📘 Running MMLU generation pipeline"

# 5️⃣ Launch the job
srun -p mllm_safety --quotatype=spot --gres=gpu:1 --time=03:00:00 \
    python MMLU_editflow_eval.py \
    --model_name_or_path "$model_path" \
    --tau 0.002 \
    --mask_length ${mask_length} \
    --field "college_mathematics" \
    --seed 7070 \