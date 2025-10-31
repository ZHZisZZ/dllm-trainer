# Generative BERT

<!-- ### 🤗 BERT Chat Checkpoints
* `ModernBERT-large-chat-v1`
* `ModernBERT-base-chat-v1`

This directory provides (1) toy examples of pretraining and SFTing any BERT model and (2) the exact training / inference / evaluation scripts for the two BERT Chat checkpoints, two [ModernBERT-chat](https://arxiv.org/abs/2412.13663) models finetuned for instruction following.

See [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) **Report**](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Generative-BERT--VmlldzoxNDg0MzExNg) for experimental results, lessons learned and more reproduction details.

<details>
<summary>🎬 Click to show Chat Demo</summary>

<p align="center" style="margin-top: 15px;">
    <img src="/examples/bert/assets/chat.gif" alt="chat" width="70%">
</p>
<p align="center">
  <em>
    Chat with <a href="[TODO]"><code>ModernBERT-large-chat-v1</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
  </em>
</p>
</details> -->

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)]([TODO])
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Generative-BERT--VmlldzoxNDg0MzExNg)

This directory provides two key sets of resources:

1.  **Toy Examples ([Warmup](#warmup)):** Scripts for pretraining and SFTing any BERT-style model on small datasets to generate text.
2.  **Official Scripts ([BERT Chat](#bert-chat)):** The exact training, inference, and evaluation scripts used to create the `ModernBERT-large-chat-v1` and `ModernBERT-base-chat-v1` checkpoints 🤗, two BERTs finetuned as Chatbots. For a deep dive into experimental results, lessons learned, and more reproduction details, please see our full [![report](https://img.shields.io/badge/W&B-black?logo=weightsandbiases) BERT Chat Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Generative-BERT--VmlldzoxNDg0MzExNg).

<details>
<summary>🎬 **Click to show Chat Demo**</summary>

<p align="center" style="margin-top: 15px;">
    <img src="/examples/bert/assets/chat.gif" alt="chat" width="70%">
</p>
<p align="center">
  <em>
    Chat with <a href="[TODO]"><code>ModernBERT-large-chat-v1</code></a>. See <a href="/examples/bert/README.md/#inference">Inference</a> for details.
  </em>
</p>

</details>


<!-- ## Table of Contents
- [Files overview](#files-overview)
- [Warmup](#warmup)
- [BERT Chat](#bert-chat) -->

## Files overview
```
# example entry points for training / inference
examples/bert
├── chat.py                         # Interactive inference example
├── generate.py                     # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```


## Warmup

In this section, we show toy examples of pretraining and SFTing [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on small datasets to generate text.
You can use any BERT model instead for example, by `--model_name_or_path "FacebookAI/roberta-large"`.

### Pretrain

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/bert/pt.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --max_length 128 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tiny-shakespeare"
```

To run inference with the model:
```shell
# just press enter (empty prompt) if you want the model to generate text from scratch 
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/tiny-shakespeare/checkpoint-final" \
    --chat False --remasking "random" --steps 128 --max_new_tokens 128
```

### SFT

To train [`ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset, run:
```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/alpaca"
```

To run inference with the model:
```shell
python -u examples/bert/chat.py \
    --model_name_or_path "models/ModernBERT-large/alpaca/checkpoint-final" --chat True
```

## BERT Chat

### Training

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-base" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-base/tulu-3-smoltalk/epochs-10-bs-384-len-1024"

accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/bert/sft.py \
    --model_name_or_path "answerdotai/ModernBERT-large" \
    --dataset_args "allenai/tulu-3-sft-mixture|HuggingFaceTB/smoltalk" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 48 \
    --per_device_eval_batch_size 48 \
    --save_steps 0.1 \
    --output_dir "models/ModernBERT-large/tulu-3-smoltalk/epochs-10-bs-384-len-1024"
```

### Inference

To run inference with the model:
```shell
python -u examples/bert/chat.py --model_name_or_path "[TODO]" --chat True
```

### Evaluation
[TODO]
