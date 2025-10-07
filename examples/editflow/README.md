# Edit Flows

> **Reference**
> 📄 Paper: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 

This directory provides an educational reference for training EditFlow models. It demonstrates how to adapt open-weight DLLMs—such as [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487)—to support *insertion*, *deletion*, beyond the standard *substitution*(`mask`->`tokens`) operations. It also includes examples for training (pretraining and finetuning) EditFlow models from scratch.

> [!NOTE]
> - While both examples for LLaDA and Dream are available. This README will focus on examples of adapting LLaDA ([`adapt_llada.py`](/examples/editflow/adapt_llada.py)) and reusing the LLaDA architecture for training from scratch ([`pt_llada.py`](/examples/editflow/pt_llada.py) -> [`sft_llada.py`](/examples/editflow/sft_llada.py)).
> - While custom `x0` distributions are supported via `EditFlowCollator`, this README uses a fixed-length scheme (128 mask tokens). The trained model generates text by replacing masks, deleting extras, and inserting tokens as needed. To switch the default `x0` distribution (e.g., to empty sequences so generation is insertion-only at inference), pass `--x0_sampler "sample_x0_empty"` when launching the job.


## Setup notes
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` for your cluster:
> ```diff
> - #SBATCH --partition=mllm_safety # Note: adjust this for your cluster
> - #SBATCH --quotatype=spot        # Note: adjust this for your cluster
> + #SBATCH --partition=YOUR_PARTITION
> + #SBATCH --quotatype=YOUR_QUOTATYPE
> ```
>

##  Files overview
```
dllm/pipelines/editflow
├── __init__.py                 # Package initialization
├── models
│   ├── dream
│   │   └── modelling_dream.py  # EditFlowDream: architecture based on Dream
│   └── llada
│       └── modelling_llada.py  # EditFlowLLaDA: architecture based on LLaDA
├── trainer.py
└── utils.py

# example entry point for training / sampling
examples/editflow
├── adapt_dream.py              # Example of adapting Dream for EditFlow directly
├── adapt_llada.py              # Example of adapting LLaDA for EditFlow directly
├── generate.py                 # Generation example
├── pt_dream.py                 # EditFlowDream pretraining example
├── pt_llada.py                 # EditFlowLLaDA pretraining example
├── pt.py                       # Pretraining function
├── README.md                   # Documentation (you are here)
├── sft_dream.py                # EditFlowDream SFT example
├── sft_llada.py                # EditFlowLLaDA SFT example
└── sft.py                      # Supervised finetuning function
```

## Adapting [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) to support *insertion* and *deletion*

The original LLaDA model generated text by iteratively substituting the given `mask` tokens to real tokens. 

<!-- <div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="https://github.com/ML-GSAI/LLaDA/blob/main/imgs/example_gradio.gif" style="width: 80%" />
</div> -->
<p align="center">
  <img src="https://github.com/ML-GSAI/LLaDA/blob/main/imgs/example_gradio.gif" alt="LLaDA demo" width="80%">
</p>
<p align="center"><em>Figure: Example Gradio demo for LLaDA.</em></p>

However, LLaDA natively supports only substitution. This example shows how to adapt it so that, during decoding, the model can not only replace fixed-length masks (e.g., 128 tokens) with real text but also insert new tokens and delete unnecessary masks adaptively:

```shell
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml examples/editflow/adapt_llada.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lm_head_key "model.transformer.ff_out" \
    --init_editflow_from_src True \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture[train:10000,test:1000]" \
    --x0_sampler "sample_x0_masks" \
    --max_length 1024 \ 
    --num_train_epochs 4
```

If you are using slurm and want to train across, for example, two nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/editflow/adapt_llada.py" \
    --script_args '
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lm_head_key "model.transformer.ff_out" \
    --init_editflow_from_src True \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture[train:10000,test:1000]" \
    --x0_sampler "sample_x0_masks" \
    --max_length 1024 \ 
    --num_train_epochs 4
    '
```

After training, you can use the generate scripts to provide a visualized decoding trace to see how the model performs *insertion* and *deletion* beyond regular mask *substitutions*.


## Pretraining & Finetuning from scratch
You can also train an EditFlow model from scratch (pretrain → SFT) without adapting an existing DLLM.

Pretrain on a subset of [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 256 GPUs (32x8) and DeepSpeed ZeRO-2:

```shell
sbatch --nodes=32 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/editflow/pt_llada.py" \
    --script_args '
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]" \
    --output_dir "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]" \
    --x0_sampler "sample_x0_masks" \
    --max_length 1024 \ 
    --max_steps 10000
    '
```

Finetune on a subset of [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUS and DeepSpeed ZeRO-2 for better instruction following:

```shell
# you can also run locally with `accelerate ...`
sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/editflow/sft_llada.py" \
    --script_args '
    --model_name_or_path "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir "models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0[train:10_000_000,test:10_000]" \
    --x0_sampler "sample_x0_masks" \
    --max_length 1024 \ 
    --num_train_epochs 4
    '
```

## Acknowledgement

This Edit Flows implementation is inspired by https://github.com/TheMatrixMaster/edit-flows-demo.