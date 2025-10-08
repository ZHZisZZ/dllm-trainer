# LLaDA

> **Reference**  
> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
> 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

This directory provides examples for finetuning open-weight LLaDA models, reproducing LLaDA by training from scratch on public data (pretraining & finetuning), and batch sampling for generation tasks.

## Table of Contents
- [Setup notes](#setup-notes)
- [Files overview](#files-overview)
- [Finetuning LLaDA-8B-Base](#finetuning-llada-8b-base)
- [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)

## Setup notes
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps` before submitting sbatch jobs: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
>


##  Files overview
```
# tools relevant with LLaDA
dllm/pipelines/llada
├── generate.py                     # Generation utilities
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
└── trainer.py                      # Training logic (pretraining and finetuning)

# example entry points for training / sampling
examples/llada
├── generate.py                     # Generation example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
> [!NOTE]
>  We fixed bugs that caused attention masks to not work correctly in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py).

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Finetuning [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
<!-- > [!NOTE]
> If you are using slurm, please modify [`scripts/train.slurm.sh`](/scripts/train.slurm.sh) for your cluster
> 
> ```diff
> - #SBATCH --partition=mllm_safety # Note: adjust this for your cluster
> - #SBATCH --quotatype=spot        # Note: adjust this for your cluster
> + #SBATCH --partition=YOUR_PARTITION
> + #SBATCH --quotatype=YOUR_QUOTATYPE
> ``` -->

We support training models with either DDP or DeepSpeed ZeRO-{1,2,3}. For example, to SFT [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) for instruction following on a subset of [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using DeepSpeed ZeRO-2 on 8 GPUs, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture[train:10000,test:1000]" \
    --max_length 1024 \ 
    --num_train_epochs 4
```
If you are using slurm and want to train across, for example, two nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --script_args '
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture[train:10000,test:1000]" \
    --max_length 1024 \ 
    --num_train_epochs 4
    '
```

## Pretraining & finetuning from scratch
> [!NOTE]
> This is an educational example demonstrating how to reproduce LLaDA pretraining and finetuning on public data. We do not guarantee performance comparable to the official LLaDA models.

Pretrain on a subset of [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 256 GPUs (32x8) and DeepSpeed ZeRO-2:
```shell
sbatch --nodes=32 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/pt.py" \
    --script_args '
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]" \
    --output_dir "models/LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]" \
    --max_length 1024 \ 
    --max_steps 10000
    '
```
Finetune on a subset of [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUS and DeepSpeed ZeRO-2 for better instruction following:
```shell
# you can also run locally with `accelerate ...`
sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --script_args '
    --model_name_or_path "models/LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture[train:10000,test:1000]" \
    --max_length 1024 \ 
    --num_train_epochs 4
    '
```

## Sampling
We support batch sampling for both continuation given prompts and infilling generation, where masked tokens are interleaved with provided text.
See [`examples/llada/generate.py`](https://github.com/ZHZisZZ/dllm/blob/main/examples/llada/generate.py) for a full sampling example.
```shell
python examples/llada/generate.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
