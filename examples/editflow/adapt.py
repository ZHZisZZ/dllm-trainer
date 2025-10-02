"""
Local users
------------
- 1 GPU:
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/editflow/adapt.py

- 1 GPU (4bit quant, LoRA) & Weight merging:
    # train
    accelerate launch \
        --config_file scripts/accelerate_configs/single_gpu.yaml \
        examples/editflow/adapt.py \
        --load_in_4bit True --lora True

    # merge lora weights
    python dllm_trainer/tools/merge_peft_adapter.py \
        --adapter_model_name_or_path models/Dream-7B-SFT/checkpoint-final \
        --output_model_name_or_path models/Dream-7B-SFT/checkpoint-final-merged \
        --dtype bf16
    
- 8 GPUs (DeepSpeed ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        examples/editflow/adapt.py

Slurm users
# Note: run `mkdir logs` before running sbatch; and adjust 
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 GPU:
    sbatch scripts/train.slurm.sh \
        --accelerate_config "single_gpu" \
        --script_path "examples/editflow/adapt.py"

- 8 GPUs (DeepSpeed ZeRO-2):
    sbatch scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/adapt.py"

- 2 Nodes, 16 GPUs (DeepSpeed ZeRO-2):
    sbatch --nodes=2 scripts/train.slurm.sh \
        --accelerate_config "deepspeed_zero2" \
        --script_path "examples/editflow/adapt.py"
"""
import os
import functools
from dataclasses import dataclass
from collections import OrderedDict

import torch
import transformers
import accelerate

import dllm
from dllm.pipelines import editflow, dream


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"

@dataclass
class DataArguments:
    dataset_args: str = "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"

@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/EditFlow-Dream-Instruct-7B/tulu-3-sft-mixture[train:10000,test:1000]"
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    # others (editflow specific training params)
    scheduler_cls: str = "LinearKappaScheduler"
    normalize_per_position: bool = True
    max_w: float = 20
    x0_sampler: str = "sample_x0_mixture" # sample_x0_masks, sample_x0_empty, sample_x0_noisy, sample_x0_mixture
    mask_prompt_loss: bool = True


def init_editflow_from_dream(ef_model, dream_model, lm_head_key="lm_head", verbose=True):
    """
    Initialize an EditFlowDreamModel (ef_model) from a pretrained DreamModel (dream_model).

    - Copies all matching backbone params.
    - Duplicates Dream lm_head -> ef_model.sub_logits and ef_model.ins_logits.
    - Leaves new rate heads (sub_rate/ins_rate/del_rate) as-is (random init).
    - Returns (missing_keys, unexpected_keys) from load_state_dict(strict=False).

    Args:
        ef_model:      EditFlowDreamModel instance (target).
        dream_model:   DreamModel instance (source).
        lm_head_key:   Base key name for DreamModel's LM head (default: "lm_head").
        verbose:       If True, prints a short load report.

    Example:
        dream = DreamModel.from_pretrained(path)
        ef    = EditFlowDreamModel.from_config(cfg)
        init_editflow_from_dream(ef, dream)
    """
    src_sd = dream_model.state_dict()
    tgt_sd = ef_model.state_dict()
    new_sd = OrderedDict()

    # 1) copy matching tensors (same key & shape)
    for k, v in src_sd.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            new_sd[k] = v

    # 2) duplicate lm_head -> sub_logits & ins_logits (weight + optional bias)
    lm_w = f"{lm_head_key}.weight"
    lm_b = f"{lm_head_key}.bias"

    if lm_w in src_sd:
        if "sub_logits.weight" in tgt_sd:
            new_sd["sub_logits.weight"] = src_sd[lm_w]
        if "ins_logits.weight" in tgt_sd:
            new_sd["ins_logits.weight"] = src_sd[lm_w]

    if lm_b in src_sd:
        if "sub_logits.bias" in tgt_sd:
            new_sd["sub_logits.bias"] = src_sd[lm_b]
        if "ins_logits.bias" in tgt_sd:
            new_sd["ins_logits.bias"] = src_sd[lm_b]

    # 3) load non-strictly so new heads can stay randomly initialized
    missing, unexpected = ef_model.load_state_dict(new_sd, strict=False)

    if verbose:
        dllm.utils.print_main(f"[EditFlow init] Copied {len(new_sd)} tensors from DreamModel.")
        if missing:
            dllm.utils.print_main("  Missing (expected for new rate heads, etc.):")
            for k in missing:
                dllm.utils.print_main("   -", k)
        if unexpected:
            dllm.utils.print_main("  Unexpected (check key names):")
            for k in unexpected:
                dllm.utils.print_main("   -", k)

    return missing, unexpected


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        dllm.utils.DataArguments, 
        TrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = []
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)

    # ----- Load base Dream and initialize EditFlowDream ---------------------------
    model_name_or_path = dllm.utils.resolve_with_base_env(
        model_args.model_name_or_path, "BASE_MODELS_DIR")

    # Load Dream config & weights (bf16 on CUDA)
    dream_cfg = editflow.EditFlowDreamConfig.from_pretrained(model_name_or_path)
    dream_model = dream.DreamModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda")

    # Create EditFlow model (bf16 init on CUDA), then copy/clone from Dream
    with dllm.utils.init_on("cuda", torch.bfloat16):
        model = editflow.EditFlowDreamModel(dream_cfg)

    # Initialize EditFlow from Dream: copies backbone & clones lm_head
    init_editflow_from_dream(model, dream_model)
    del dream_model

    def _no_flops(*args, **kwargs): return 0.0
    model.floating_point_ops = _no_flops

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model, model_args)

    # ----- Dataset ----------------------------------------------------------------
    # Build emulated pretraining samples from SFT chats:
    # - `input_ids`` = prompt + response
    # - `prompt_len` marks the prompt span to EXCLUDE from loss.
    #   (Remove prompt_len to train on all tokens—if so, ensure a BOS is prepended.)
    def train_map_fn(
        row, 
        tokenizer: transformers.PreTrainedTokenizer, 
        mask_prompt_loss: bool = True, 
    ) -> dict:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_response_tokens = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=False,
        )
        # overwrite "<|im_end|>\n" to "<|im_end|><|endoftext|>"
        # TODO: delete this after overwriting chat template
        prompt_response_tokens[-1] = tokenizer.eos_token_id
        if mask_prompt_loss:
            return {
                "input_ids": prompt_response_tokens,
                "prompt_len": len(prompt_tokens), # ! Note: remove this to train on all "input_ids"
            }
        else:
            # When training on all tokens, prepend a BOS token (if missing)
            # so the model can insert to the left of the very first token.
            if prompt_response_tokens[0] != tokenizer.bos_token_id:
                prompt_response_tokens = [tokenizer.bos_token_id] + prompt_response_tokens
            return {"input_ids": prompt_response_tokens}

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(
            functools.partial(
                train_map_fn, 
                tokenizer=tokenizer,
            ), 
            num_proc=data_args.num_proc,
        )
        dataset = dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
        )

    # ----- Training --------------------------------------------------------------
    trainer = editflow.EditFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=editflow.utils.EditFlowCollator(tokenizer=tokenizer, x0_sampler=training_args.x0_sampler),
        scheduler=dllm.utils.schedulers.make_kappa_scheduler(training_args.scheduler_cls),
        normalize_per_position=training_args.normalize_per_position,
        max_w=training_args.max_w,
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    train()
