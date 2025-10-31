import os
from functools import partial
from dataclasses import dataclass, field

import transformers
import accelerate

import dllm
from dllm.pipelines import editflow


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = None  # overwrite this


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = None  # overwrite this
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    # EditFlow specific args
    scheduler_cls: str = field(
        default="LinearKappaScheduler",
        metadata={
            "help": (
                "The scheduler class controlling κ(t). "
                "Available options: see `dllm/utils/schedulers/kappa.py`"
            )
        },
    )
    normalize_per_position: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the loss per position."},
    )
    max_w: float = field(
        default=20.0,
        metadata={"help": "The maximum weight (κ'(t) / (1 - κ(t))) for the loss."},
    )
    x0_sampler: str = field(
        default="masks[length:128]",
        metadata={
            "help": (
                "Choose the x0 sampler. "
                "Available options: see `dllm/pipelines/editflow/utils.py`"
            )
        },
    )


def sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    # - `input_ids`` = prompt + response
    # - `prompt_len` marks the prompt span to EXCLUDE from loss.
    #   (Remove prompt_len to train on all tokens—if so, ensure a BOS is prepended.)
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )
    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        return {
            "input_ids": prompt_response_tokens,
            "prompt_len": len(prompt_tokens),
        }
    else:
        # When training on all tokens, prepend a BOS token (if missing)
        # so the model can insert to the left of the very first token.
        if prompt_response_tokens[0] != tokenizer.bos_token_id:
            prompt_response_tokens = [
                tokenizer.bos_token_id
            ] + prompt_response_tokens
        return {"input_ids": prompt_response_tokens}


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    model: transformers.PreTrainedModel | None = None,
):
    # necessary when batch does not contain "labels" field
    training_args.label_names = []
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # # ----- Load EditFlow Model --------------------------------------------------
    if not model:
        model = dllm.utils.get_model(model_args=model_args)

    def _no_flops(*args, **kwargs):
        return 0.0

    model.floating_point_ops = _no_flops

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(map_fn, num_proc=data_args.num_proc)
        # truncate / filter long sequences if needed
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Training --------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("start training...")
    trainer = editflow.EditFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=editflow.utils.EditFlowCollator(
            tokenizer=tokenizer, x0_sampler=training_args.x0_sampler
        ),
        scheduler=dllm.core.schedulers.make_kappa_scheduler(
            training_args.scheduler_cls
        ),
        normalize_per_position=training_args.normalize_per_position,
        max_w=training_args.max_w,
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )
