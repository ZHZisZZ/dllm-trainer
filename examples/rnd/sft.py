"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/rnd/sft.py

- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/rnd/sft.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch --gres=gpu:1 scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/rnd/sft.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/rnd/sft.py"
"""

import os
from dataclasses import dataclass, field

import transformers
import accelerate
import peft
import datasets

import dllm
from dllm.pipelines import rnd


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "radicalnumerics/RND1-Base-0910"
    moe_backend: str = "hf"
    attn_implementation: str = "sdpa"

@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "HuggingFaceTB/smoltalk[train:10000,test:1000]"
    truncation: str = "right"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/RND1-SFT-0910/smoltalk[train:10000,test:1000]"
    # rnd specific
    group_by_length: bool = True
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    freeze_gate: bool = field(
        default=True,
        metadata={"help": "If True, freeze routing gate parameters (e.g., MoE router/gating layers)."},
    )
    freeze_embedding: bool = field(
        default=False,
        metadata={"help": "If True, freeze embedding parameters."},
    )


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        moe_backend=model_args.moe_backend,
        attn_implementation=model_args.attn_implementation,
    )
    model = dllm.utils.get_model(model_args=model_args, config=config)
    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    # ----- Optionally freeze modules ----------------------------------------------
    if not isinstance(model, peft.PeftModel):
        if getattr(training_args, "freeze_gate", False):
            for n, m in model.named_modules():
                if n.endswith(".gate"):  # only router gate, not gate_proj
                    for p in m.parameters(recurse=False):
                        p.requires_grad_(False)

        if getattr(training_args, "freeze_embedding", False):
            # model.model.embed_tokens.requires_grad_(False)
            model.model.embed_tokens.weight.requires_grad_(False)

    # ----- Dataset ----------------------------------------------------------------
    def sft_map_fn(row) -> dict:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False
        )
        labels = prompt_response_tokens.copy()
        if training_args.mask_prompt_loss:
            # use -100 in labels to indicate positions where tokens should not be masked
            # and loss is ignored; all other positions match `input_ids`
            labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        else:
            # When training on all tokens, prepend a BOS token (if missing)
            # so the model can make predictions for the first mask token.
            if prompt_response_tokens[0] != tokenizer.bos_token_id:
                bos = [tokenizer.bos_token_id]
                prompt_response_tokens = bos + prompt_response_tokens
                prompt_tokens = bos + prompt_tokens
                labels = bos + labels
            labels[0] = -100  # ignore loss on the BOS token
        # `prompt_len` helps `post_process_dataset` truncate long sequences properly
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            # "attention_mask": [1.0] * len(prompt_response_tokens),
            "prompt_len": len(prompt_tokens),
        }

    if not data_args.load_from_disk:
        with accelerate.PartialState().local_main_process_first():
            dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
            dataset = dataset.map(sft_map_fn, num_proc=data_args.num_proc)
            # truncate / filter long sequences if needed
            dataset = dllm.utils.post_process_dataset(dataset, data_args)
    else:
        from datasets import disable_caching; disable_caching()
        dataset = datasets.load_from_disk(data_args.dataset_args)
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    @dataclass
    class RNDSFTCollator(transformers.DataCollatorForSeq2Seq):
        def __call__(self, features, return_tensors=None):
            outputs = super().__call__(features, return_tensors)
            # RND is finetuned on padding <eos_token>
            outputs.pop("attention_mask")
            # temp fix here (`group_by_length=True` leads to shape mismatch)
            # clip seq_len (second dim) to the same for outputs `input_ids, labels`
            import torch
            keys_to_clip = [k for k in ("input_ids", "labels") if k in outputs]
            if keys_to_clip:
                # Get smallest seq_len to avoid out-of-bounds
                min_len = min(outputs[k].size(1) for k in keys_to_clip if isinstance(outputs[k], torch.Tensor))
                for k in keys_to_clip:
                    t = outputs[k]
                    if isinstance(t, torch.Tensor) and t.size(1) != min_len:
                        outputs[k] = t[:, :min_len]
            return outputs

    trainer = rnd.RNDTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=RNDSFTCollator(
            tokenizer,
            # pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,  # RND is finetuned on padding <eos_token>
        ),
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
