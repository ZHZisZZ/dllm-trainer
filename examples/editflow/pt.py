import os
import functools
from dataclasses import dataclass

import torch
import transformers
import accelerate

import dllm
from dllm.pipelines import editflow


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = None  # TODO: overwrite this
    lm_head_key: str = None  # TODO: overwrite this if `init_editflow_from_src` = True
    init_editflow_from_src: bool = True


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = (
        "dataset_name_or_path=mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    )
    truncation: str = "right"
    max_length: int = 2048


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = None  # TODO: overwrite this
    learning_rate: float = 3e-4
    max_steps: int = 10_000
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    eval_steps: float = 0.1
    save_steps: float = 0.1
    # others (editflow specific training params)
    scheduler_cls: str = "LinearKappaScheduler"
    normalize_per_position: bool = True
    max_w: float = 20
    x0_sampler: str = (
        "sample_x0_masks"  # sample_x0_masks, sample_x0_empty, sample_x0_noisy, sample_x0_mixture
    )


def train(
    model_args: ModelArguments,
    data_args: dllm.utils.DataArguments,
    training_args: TrainingArguments,
    ef_config_cls: type[transformers.PretrainedConfig],
):
    training_args.label_names = (
        []
    )  # necessary when batch does not contain "labels" field
    training_args.remove_unused_columns = (
        False  # necessary when batch contains customized fields
    )
    training_args.accelerator_config.dispatch_batches = (
        False  # necessary for streaming dataset
    )
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Load base Model and initialize EditFlow Model ---------------------------
    # Create EditFlow model (bf16 init on CUDA)
    ef_cfg = ef_config_cls.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(ef_cfg, torch_dtype=torch.bfloat16)
    if model_args.init_editflow_from_src:
        # Load src model config & weights (bf16 on CUDA) for intializing EditFlow model
        src_model = dllm.utils.get_model(model_args)
        # Initialize EditFlow model from the src model: copies backbone & clones lm_head
        editflow.utils.init_editflow_from_src(
            model, src_model, lm_head_key=model_args.lm_head_key
        )
        del src_model

    def _no_flops(*args, **kwargs):
        return 0.0

    model.floating_point_ops = _no_flops

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    def pt_map_fn(
        row,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> dict:
        input_ids = tokenizer.encode(row["text"])
        if input_ids[0] != tokenizer.bos_token_id:
            input_ids = [tokenizer.bos_token_id] + input_ids
        return {"input_ids": input_ids, "labels": input_ids}

    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_pt_dataset(data_args.dataset_args)
        dataset = dataset.map(functools.partial(pt_map_fn, tokenizer=tokenizer))
        dataset = dllm.utils.post_process_dataset_streaming(
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
