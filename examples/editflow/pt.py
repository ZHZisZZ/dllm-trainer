import os
import functools
from dataclasses import dataclass, field

import torch
import transformers
import accelerate

import dllm
from dllm.pipelines import editflow


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = None  # overwrite this
    lm_head_key: str = field(
        default=None,
        metadata={
            "help": (
                "The key to the `lm_head` in the source model for initializing operation heads in the EditFlow model. "
                "Overwrite this when `init_editflow_from_src` = True"
            )
        },
    )
    init_editflow_from_src: bool = field(
        default=True,
        metadata={
            "help": "Whether to initialize EditFlow model from the source model."
        },
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "mlfoundations/dclm-baseline-1.0[train:10_000_000,test:10_000]"
    truncation: str = "right"


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = None  # overwrite this
    learning_rate: float = 3e-4
    max_steps: int = 2_000
    per_device_train_batch_size: int = 3
    gradient_accumulation_steps: int = 4
    eval_steps: float = 0.05
    save_steps: float = 0.05
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
        metadata={
            "help": (
                "Choose the x0 sampler. "
                "Available options: see `dllm/pipelines/editflow/utils.py`"
            )
        },
    )
    x0_sampler: str = field(
        default="masks[length:128]",
        metadata={"help": "The x0 sampler to use. Default to 128 mask tokens."},
    )


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    ef_config_cls: type[transformers.PretrainedConfig],
):
    # necessary when batch does not contain "labels" field
    training_args.label_names = []
    # necessary when batch contains customized fields
    training_args.remove_unused_columns = False
    # necessary for streaming dataset
    training_args.accelerator_config.dispatch_batches = False
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(training_args)

    # ----- Load base Model and initialize EditFlow Model ---------------------------
    # Create EditFlow model (bf16 init on CUDA)
    ef_cfg = ef_config_cls.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(ef_cfg, dtype=torch.bfloat16)
    if model_args.init_editflow_from_src:
        # Load src model config & weights (bf16 on CUDA) for intializing EditFlow model
        src_model = dllm.utils.get_model(model_args=model_args)
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
