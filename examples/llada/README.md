# LLaDA

> **Reference**  
> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
> 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

This directory provides examples for (1) finetuning open-weight LLaDA models, (2) pretraining from scratch on public data, (3) interactive inference and (4) evaluation.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    <!-- - [Finetuning LLaDA-8B-Base](#finetuning-llada-8b-base)
    - [Pretraining from scratch](#pretraining-from-scratch) -->
- [Inference](#inference)
- [Evaluation](#evaluation)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
>


##  Files overview
```
# tools relevant with LLaDA
dllm/pipelines/llada
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
├── generator.py                    # Inference logic
└── trainer.py                      # Training logic (pretraining and finetuning)

# example entry points for training / inference
examples/llada
├── chat.py                         # Interactive inference example
├── generate.py                     # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
<!-- > [!NOTE] -->
<!-- >  - We fixed attention mask bugs in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py). We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
>  - We fixed bugs in `chat_template` and assign `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves. -->

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Training

<!-- > [!NOTE]
> Here are some useful tips for training:
> - Use a subset of data: `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"`; 
> 
> - Concatenate datasets: `--dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk"`;
>
> - Train with LoRA and 4bit quantization: `--load_in_4bit True --lora True`. -->

### Finetuning

For example, to SFT [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

<!-- **Reproducing [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)**. Though LLaDA is trained on proprietary data, we tried our best to reproduce LLaDA-8B-Instruct by finetuning LLaDA-8B-Base using our training pipeline on public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->

#### Reproducing [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
Though LLaDA is trained on proprietary data, we tried our best to reproduce LLaDA-8B-Instruct by finetuning LLaDA-8B-Base using our training pipeline on public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture):

```shell
# preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/llada/tulu-3-sft-mixture" \
    --num_proc 64

# train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "data/sft/llada/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --output_dir "models/LLaDA-8B-SFT-tulu3-fsdp-bs4-len2048-ep5-lr1e-5" \
    --max_length 2048 \
    --truncation "right" \
    --group_by_length True \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --eval_on_start False \
    --eval_steps 0.1 \
    --save_steps 0.05
```
Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions.
[TODO]


### Pretraining
<!-- > [!NOTE] 
> 
> This is an educational example demonstrating how to reproduce LLaDA pretraining and finetuning on public data. We do not guarantee performance comparable to the official LLaDA models. -->

Pretrain on [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/LLaDA-8B-PT/dclm-baseline-1.0" \
    --max_length 1024 \ 
    --max_steps 2000 \
    --learning_rate 3e-4
```

## Inference
We support batch inference for standard generation and infilling:
See [`examples/llada/generate.py`](/examples/llada/generate.py) for a full example.
```shell
python examples/llada/generate.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
We also support interactive multi-turn dialogue with visualization:
<!-- See [`examples/llada/chat.py`](/examples/llada/chat.py) for a full example. -->
```shell
python examples/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

## Evaluation
[TODO]
