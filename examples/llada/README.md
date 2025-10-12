# LLaDA

> **Reference**  
> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
> 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

This directory provides examples for finetuning open-weight LLaDA models, reproducing LLaDA by training from scratch on public data (pretraining & finetuning), and batch sampling for generation tasks.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    - [Finetuning LLaDA-8B-Base](#finetuning-llada-8b-base)
    - [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)

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
>  We fixed attention mask bugs in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py). We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
>  We fixed bugs in `chat_template` and standardize `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves.

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Training

> [!NOTE]
> Use `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"` to train / eval only on a subset; 
> 
> Use `--dataset_args "allenai/tulu-3-sft-mixture | OpenCoder-LLM/opc-sft-stage2[name:educational_instruct]"` to concatenate datasets.

### Finetuning [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
We support training models with either DDP or DeepSpeed ZeRO-{1,2,3}. For example, to SFT [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) for instruction following on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using DeepSpeed ZeRO-2 on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, four nodes (32 GPUs total), run:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

### Pretraining & finetuning from scratch
> [!NOTE]
> This is an educational example demonstrating how to reproduce LLaDA pretraining and finetuning on public data. We do not guarantee performance comparable to the official LLaDA models.

Pretrain on [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 256 GPUs (32x8) and DeepSpeed ZeRO-2:
```shell
sbatch --nodes=32 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/LLaDA-8B-PT/dclm-baseline-1.0" \
    --max_length 1024 \ 
    --max_steps 2000 \
    --learning_rate 3e-4
```
Finetune on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUs and DeepSpeed ZeRO-2 for better instruction following:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "models/LLaDA-8B-PT/dclm-baseline-1.0/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

## Sampling
We support batch sampling for standard generation and infilling generation.
See [`examples/llada/generate.py`](/examples/llada/generate.py) for a full example.
```shell
python examples/llada/generate.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
