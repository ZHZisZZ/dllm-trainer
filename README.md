## Overview
A lightweight framework for training diffusion language models, built on top of the [🤗 Transformers](https://github.com/huggingface/transformers) `Trainer`. It currently supports SFT (*deepspeed-zero{1,2,3}, multinode training, LoRA*) and batch sampling (*continuation, fill-in-blanks*) for [LLaDA / LLaDA-MoE](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487). 

## Setup
```bash
# create and activate conda environment
conda create -n dllm python=3.10 -y
conda activate dllm

# install pytorch with CUDA 11.8 (other pytorch/cuda versions should also work)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt

# install dllm package
pip install -e .
```

## Quick Start

<details>
<summary>LLaDA / LLaDA-MoE: SFT and Sampling</summary>

### `SFT`
Basic usage of [`LLaDATrainer`](https://github.com/ZHZisZZ/dllm/blob/main/dllm/pipelines/llada/trainer.py#L12). See [`examples/llada/sft.py`](https://github.com/ZHZisZZ/dllm/blob/main/examples/llada/sft.py) for a complete example.
```python
import transformers

import dllm

model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model = dllm.utils.get_model(model_args, training_args)
tokenizer = dllm.utils.get_tokenizer(model_args, model)
dataset = "..."

# ----- Training --------------------------------------------------------------
trainer = dllm.pipelines.llada.LLaDATrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id, # LLaDA is trained on padding <eos_token>
    )
)
trainer.train()
```

> **Notes (LLaDA-MoE only):**  
> For MoE checkpoints, overwrite `config.json` with the following `model_type` and `auto_map`:  
> ```json
> {
>   "model_type": "lladamoe",
>   "auto_map": {
>     "AutoConfig": "configuration_lladamoe.LLaDAMoEConfig",
>     "AutoModel": "modeling_lladamoe.LLaDAMoEModelLM",
>     "AutoModelForCausalLM": "modeling_lladamoe.LLaDAMoEModelLM",
>   }
> }
> ```


### `Sampling`
See [`examples/llada/generate.py`](https://github.com/ZHZisZZ/dllm/blob/main/examples/llada/generate.py) for a complete example of batch sampling (continuation and fill_in_blanks).

</details>

<details>
<summary>Dream: SFT and Sampling</summary>

### `SFT`

Basic usage of [`DreamTrainer`](https://github.com/ZHZisZZ/dllm/blob/main/dllm/pipelines/dream/trainer.py#L39). See [`examples/dream/sft.py`](https://github.com/ZHZisZZ/dllm/blob/main/examples/dream/sft.py) for a complete example.
```python
import transformers

import dllm

model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model = dllm.utils.get_model(model_args, training_args)
tokenizer = dllm.utils.get_tokenizer(model_args, model)
dataset = "..."

# ----- Training --------------------------------------------------------------
trainer = dllm.pipelines.dream.DreamTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
        label_pad_token_id=-100 # padding tokens do not count in loss
    )
)
trainer.train()
```


### `Sampling`
See [`examples/dream/generate.py`](https://github.com/ZHZisZZ/dllm/blob/main/examples/dream/generate.py) for a complete example of batch sampling (continuation and fill_in_blanks).

</details>


## TODO
- [ ] **Pretraining scripts**: Add sample scripts for pretraining (trainer already supports both pretraining and finetuning; main difference is data preprocessing).  

- [ ] **Support for additional diffusion LLMs**: Add support beyond LLaDA / LLaDA-MoE / Dream.  

- [ ] **Support for more finetuning algorithms**: Implement and benchmark other finetuning methods.


## Citation
```
@misc{dllm,
    author = {Zhanhui Zhou and Lingjie Chen},
    title = {dllm: Diffusion Large Language Models Training},
    howpublished = {https://github.com/ZHZisZZ/dllm},
    note = {Accessed: 2025-09-21},
    year = {2025}
}
```
