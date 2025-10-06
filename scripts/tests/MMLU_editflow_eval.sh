#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Parse model choice + mask length (defaults)
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

# 4️⃣ Dataset base directory containing all MMLU fields
dataset_base="/mnt/lustrenew/mllm_safety-shared/datasets/huggingface/cais/mmlu"

# 5️⃣ Output directory for results
output_dir="./mmlu_results/${choice}_mask${mask_length}"
model_name=$(basename "$(dirname "${model_path}")")

mkdir -p "${output_dir}"

echo "🚀 Using model: ${model_path}"
echo "📘 Running MMLU evaluation over all fields"
echo "💾 Results will be saved to: ${output_dir}"


# 6️⃣ Loop over all subdirectories (fields) and skip unwanted files
for field_path in "${dataset_base}"/*; do
    field_name=$(basename "${field_path}")

    # Skip unwanted files
    if [[ "${field_name}" == *.json ]] || [[ "${field_name}" == *.tar ]] || [[ "${field_name}" == *.py ]] || [[ "${field_name}" == *.md ]]; then
        echo "⏭️ Skipping non-dataset file: ${field_name}"
        continue
    fi

    # Target result file name
    result_file="${output_dir}/${model_name}_mask${mask_length}_${field_name}.txt"
    echo $result_file

    # Skip if result already exists
    if [[ -f "${result_file}" ]]; then
        echo "⏩ Skipping ${field_name} (already exists: ${result_file})"
        continue
    fi

    echo "=============================="
    echo "📌 Evaluating field: ${field_name}"
    echo "=============================="

    srun -p mllm_safety --quotatype=spot --gres=gpu:1 --time=03:00:00 \
        python MMLU_editflow_eval.py \
        --model_name_or_path "${model_path}" \
        --tau 0.002 \
        --mask_length "${mask_length}" \
        --field "${field_name}" \
        --seed 7070 \
        --output_dir "${output_dir}"&
    sleep 0.8
done

echo "✅ All fields evaluated!"
