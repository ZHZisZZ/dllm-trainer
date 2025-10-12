# Dream

> **Reference**  
> 📄 Paper: [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
> 💻 Code: [github.com/DreamLM/Dream](https://github.com/DreamLM/Dream)

This directory provides examples for finetuning open-weight Dream models, reproducing Dream by training from scratch on public data (pretraining & finetuning), and batch sampling for generation tasks.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    - [Finetuning Dream-v0-Base-7B](#finetuning-dream-v0-base-7b)
    - [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps` before submitting sbatch jobs: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>


##  Files overview
```
# tools relevant with Dream
dllm/pipelines/dream
├── generate.py                     # Generation and diffusion sampling utilities
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_dream.py      # Dream model configuration
│   ├── generation_utils.py         # Diffusion-based generation logic
│   ├── modeling_dream.py           # Core Dream model architecture
│   └── tokenization_dream.py       # Tokenizer implementation for Dream
├── trainer.py                      # Training logic (pretraining and SFT)
└── utils.py                        # Auxiliary utilities and helper functions

# example entry points for training / sampling
examples/dream
├── generate.py                     # Sampling example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example

```
> [!NOTE]
>  We slightly modified [`modeling_dream.py`](/dllm/pipelines/dream/models/modeling_dream.py) so that the `model.forward()` supports 2-D attention masks. We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
> We fixed bugs in `chat_template` and standardize `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves.

## Training

> [!NOTE]
> Use `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"` to train/evaluate only on a subset; 
>
> Use `--dataset_args "allenai/tulu-3-sft-mixture | OpenCoder-LLM/opc-sft-stage2[name:educational_instruct]"` to concatenate datasets.

### Finetuning [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B)
Dream supports training with DDP or DeepSpeed ZeRO-{1,2,3}. For example, to SFT [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B) on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using DeepSpeed ZeRO-2 on 8 GPUs:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    examples/dream/sft.py \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/Dream-7B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, four nodes (32 GPUs total), run:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/Dream-7B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

### Pretraining & finetuning from scratch
> [!NOTE]
> This is an educational example demonstrating how to reproduce Dream pretraining and finetuning on public data. We do not guarantee performance comparable to the official Dream models.

Pretrain on [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 256 GPUs (32x8) and DeepSpeed ZeRO-2:
```shell
sbatch --nodes=32 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/dream/pt.py" \
    --model_name_or_path "Dream-org/Dream-v0-Base-7B" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/Dream-7B-PT/dclm-baseline-1.0" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 3e-4
```
Finetune on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUs and DeepSpeed ZeRO-2 for better instruction following:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/dream/sft.py" \
    --model_name_or_path "models/Dream-7B-PT/dclm-baseline-1.0/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/Dream-7B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

## Sampling
We support batch sampling for standard generation and infilling generation.
See [`examples/dream/generate.py`](/examples/dream/generate.py) for a full example.
```shell
python examples/dream/generate.py --model_name_or_path "Dream-org/Dream-v0-Base-7B"
```

