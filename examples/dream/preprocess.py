"""
python examples/dream/preprocess.py --num_proc 64 --dataset_args "HuggingFaceTB/smoltalk" --output_dir "data/sft/dream/smoltalk"
python examples/dream/preprocess.py --num_proc 64 --dataset_args "allenai/tulu-3-sft-mixture" --output_dir "data/sft/dream/tulu-3-sft-mixture"
"""
from dataclasses import dataclass
from functools import partial
import tyro

import dllm
from dllm.tools import preprocess_sft_dataset


@dataclass
class ScriptArguments(preprocess_sft_dataset.ScriptArguments):
    model_name_or_path: str = "Dream-org/Dream-v0-Base-7B"
    dataset_args: str = "HuggingFaceTB/smoltalk"  # required
    output_dir: str = "data/sft/dream/smoltalk"  # required
    remove_columns: bool = False
    num_proc: int = 32
    mask_prompt_loss: bool = True  # Mask prompt tokens in labels with -100

def main():
    from sft import sft_map_fn

    # Parse with tyro
    args = tyro.cli(ScriptArguments)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer = dllm.utils.get_tokenizer(args, model_cls=dllm.pipelines.dream.DreamModel)

    # Load your raw dataset (must contain a "messages" field per example).
    dataset = dllm.data.load_sft_dataset(args.dataset_args)

    map_fn = partial(
        sft_map_fn,
        tokenizer=tokenizer,
        mask_prompt_loss=args.mask_prompt_loss,
    )
    preprocess_sft_dataset.preprocess_sft_dataset(
        dataset=dataset, map_fn=map_fn, output_dir=args.output_dir, remove_columns=args.remove_columns, num_proc=args.num_proc)


if __name__ == "__main__":
    main()
