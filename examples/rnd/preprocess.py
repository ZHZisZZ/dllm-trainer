# """
# srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --cpus-per-task=12 --time=03:00:000

# python examples/rnd/preprocess.py --dataset_args "HuggingFaceTB/smoltalk" --output_dir "data/sft_proc/rnd/smoltalk"
# """
# import os
# from dataclasses import dataclass
# from typing import Dict, Any

# import datasets
# import transformers
# import accelerate
# import tyro

# import dllm


# # --- tyro: define dataclass for CLI args ---
# @dataclass
# class ScriptArguments:
#     """Preprocess SFT dataset (batch_size=1 only)"""
#     model_name_or_path: str = "radicalnumerics/RND1-Base-0910"
#     dataset_args: str = "HuggingFaceTB/smoltalk"  # required
#     output_dir: str = "data/sft_proc/rnd/smoltalk"  # required
#     mask_prompt_loss: bool = True  # Mask prompt tokens in labels with -100
#     # TODO: strip_cols

#     def __post_init__(self):
#         self.model_name_or_path = dllm.utils.resolve_with_base_env(
#             self.model_name_or_path, "BASE_MODELS_DIR"
#         )


# def dataset_offline_preprocess(dataset: datasets.DatasetDict, map_fn: callable, output_dir: str):
#     # Map with batch_size=1 and num_proc=1 (no batching, single process).
#     state = accelerate.PartialState()
#     with state.local_main_process_first():
#         processed = dataset.map(
#             map_fn,
#             batched=False,
#             num_proc=16,
#             load_from_cache_file=True,
#             writer_batch_size=512,
#             desc="offline preprocessing",
#         )

#         # # Keep only the three required columns to save space.
#         # keep = {"input_ids", "labels", "prompt_len"}
#         # def strip_cols(ds: datasets.Dataset) -> datasets.Dataset:
#         #     drop = [c for c in ds.column_names if c not in keep]
#         #     return ds.remove_columns(drop) if drop else ds

#         # if isinstance(processed, datasets.DatasetDict):
#         #     for split in list(processed.keys()):
#         #         processed[split] = strip_cols(processed[split])
#         # else:
#         #     processed = strip_cols(processed)

#         os.makedirs(output_dir, exist_ok=True)
#         processed.save_to_disk(output_dir)
#         print(f"[OK] Saved to: {output_dir}")



# def main():
#     # Parse with tyro
#     args = tyro.cli(ScriptArguments)

#     # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
#     tokenizer = dllm.utils.get_tokenizer(args)

#     # Load your raw dataset (must contain a "messages" field per example).
#     dataset = dllm.data.load_sft_dataset(args.dataset_args)

#     dataset_offline_preprocess(dataset=dataset, map_fn=None, output_dir=args.output_dir)


# if __name__ == "__main__":
#     main()


from functools import partial
import tyro

import dllm
from dllm.tools.preprocess_sft_dataset import ScriptArguments, preprocess_sft_dataset


def main():
    from examples.rnd.sft import sft_map_fn

    # Parse with tyro
    args = tyro.cli(ScriptArguments)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer = dllm.utils.get_tokenizer(args)

    # Load your raw dataset (must contain a "messages" field per example).
    dataset = dllm.data.load_sft_dataset(args.dataset_args)

    map_fn = partial(
        sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=args.mask_prompt_loss,
    )
    preprocess_sft_dataset(dataset=dataset, map_fn=map_fn, output_dir=args.output_dir, remove_columns=args.remove_columns)


if __name__ == "__main__":
    main()
