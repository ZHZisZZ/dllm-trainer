"""
python dllm/tools/preprocess_sft_dataset.py
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from functools import partial

import datasets
import transformers
import accelerate
import tyro

import dllm



@dataclass
class ScriptArguments:
    """Preprocess SFT dataset (batch_size=1 only)"""
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    dataset_args: str = "HuggingFaceTB/smoltalk"  # required
    output_dir: str = "data/sft/llada/smoltalk"  # required
    remove_columns: bool = False
    num_proc: int = 32
    mask_prompt_loss: bool = True  # Mask prompt tokens in labels with -100

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def preprocess_sft_dataset(
    dataset: datasets.DatasetDict, 
    map_fn: callable, 
    output_dir: str, 
    remove_columns: bool = False,
    num_proc: int = 32,
):
    # Map with batch_size=1 and num_proc=1 (no batching, single process).
    state = accelerate.PartialState()
    with state.local_main_process_first():
        processed = dataset.map(
            map_fn,
            batched=False,
            num_proc=num_proc,
            load_from_cache_file=True,
            writer_batch_size=512,
            desc="offline preprocessing",
        )

        # Keep only the three required columns to save space.
        if remove_columns:
            keep = {"input_ids", "labels", "prompt_len", "attention_mask"}
            def strip_cols(ds: datasets.Dataset) -> datasets.Dataset:
                drop = [c for c in ds.column_names if c not in keep]
                return ds.remove_columns(drop) if drop else ds

            if isinstance(processed, datasets.DatasetDict):
                for split in list(processed.keys()):
                    processed[split] = strip_cols(processed[split])
            else:
                processed = strip_cols(processed)

        os.makedirs(output_dir, exist_ok=True)
        processed.save_to_disk(output_dir)
        print(f"[OK] Saved to: {output_dir}")




def main():
    from examples.llada.sft import sft_map_fn

    # Parse with tyro
    args = tyro.cli(ScriptArguments)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer = dllm.utils.get_tokenizer(args, model_cls=dllm.pipelines.llada.LLaDAModelLM)

    # Load your raw dataset (must contain a "messages" field per example).
    dataset = dllm.data.load_sft_dataset(args.dataset_args)

    map_fn = partial(
        sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=args.mask_prompt_loss,
    )
    breakpoint()
    preprocess_sft_dataset(dataset=dataset, map_fn=map_fn, output_dir=args.output_dir, remove_columns=args.remove_columns, num_proc=args.num_proc)


if __name__ == "__main__":
    main()
