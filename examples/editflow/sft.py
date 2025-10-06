import os
import functools
from dataclasses import dataclass

import transformers
import accelerate

import dllm
from dllm.pipelines import editflow


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = None  # TODO: overwrite this
    # lm_head_key: str = None # TODO: overwrite this
    # init_editflow_from_src: bool = True


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = (
        "dataset_name_or_path=allenai/tulu-3-sft-mixture[train:10000,test:1000]"
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = None  # TODO: overwrite this
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    # others (editflow specific training params)
    scheduler_cls: str = "LinearKappaScheduler"
    normalize_per_position: bool = True
    max_w: float = 20
    x0_sampler: str = (
        "sample_x0_masks"  # sample_x0_masks, sample_x0_empty, sample_x0_noisy, sample_x0_mixture
    )
    mask_prompt_loss: bool = True


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    model: transformers.PreTrainedModel | None = None,
):
    training_args.label_names = (
        []
    )  # necessary when batch does not contain "labels" field
    training_args.remove_unused_columns = (
        False  # necessary when batch contains customized fields
    )
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # # ----- Load EditFlow Model --------------------------------------------------
    if not model:
        model = dllm.utils.get_model(model_args)

    def _no_flops(*args, **kwargs):
        return 0.0

    model.floating_point_ops = _no_flops

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)
    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, training_args=training_args)

    # ----- Dataset ----------------------------------------------------------------
    # Build emulated pretraining samples from SFT chats:
    # - `input_ids`` = prompt + response
    # - `prompt_len` marks the prompt span to EXCLUDE from loss.
    #   (Remove prompt_len to train on all tokens—if so, ensure a BOS is prepended.)
    def sft_map_fn(
        row,
        tokenizer: transformers.PreTrainedTokenizer,
        mask_prompt_loss: bool = True,
    ) -> dict:
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
                "prompt_len": len(
                    prompt_tokens
                ),  # ! Note: remove this to train on all "input_ids"
            }
        else:
            # When training on all tokens, prepend a BOS token (if missing)
            # so the model can insert to the left of the very first token.
            if prompt_response_tokens[0] != tokenizer.bos_token_id:
                prompt_response_tokens = [
                    tokenizer.bos_token_id
                ] + prompt_response_tokens
            return {"input_ids": prompt_response_tokens}

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
        dataset = dataset.map(
            functools.partial(
                sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=training_args.mask_prompt_loss,
            ),
            num_proc=data_args.num_proc,
        )
        dataset = dllm.utils.post_process_dataset(
            dataset, data_args
        )  # truncate / filter long sequences if needed

    # ----- Training --------------------------------------------------------------
    trainer = editflow.EditFlowTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=editflow.utils.EditFlowCollator(
            tokenizer=tokenizer, x0_sampler=training_args.x0_sampler
        ),
        scheduler=dllm.utils.schedulers.make_kappa_scheduler(
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
