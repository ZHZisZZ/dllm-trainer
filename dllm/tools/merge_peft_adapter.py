"""
Merge a PEFT/LoRA adapter into its base model (auto-detected from adapter_config.json).

Usage:
  python dllm_trainer/tools/merge_peft_adapter.py \
    --adapter_model_name_or_path your-org/your-lora \
    --output_model_name_or_path ./merged-model \
    --dtype bf16
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoTokenizer, HfArgumentParser

import dllm  # so that no need to trust_remote_code

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float}


@dataclass
class ScriptArguments:
    adapter_model_name_or_path: str | None = field(
        default=None, metadata={"help": "Adapter repo or local path"}
    )
    output_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Where to save the merged model (folder or repo id)"},
    )
    dtype: str | None = field(default="fp16", metadata={"help": "fp16|bf16|fp32"})
    push_to_hub: bool | None = field(
        default=False, metadata={"help": "Push merged weights to the Hub"}
    )
    # Optional override if adapter config lacks base info:
    base_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Override base model if adapter config lacks it"},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    assert args.adapter_model_name_or_path, "please provide the adapter (repo or path)"
    assert args.output_model_name_or_path, "please provide output_model_name_or_path"
    assert args.dtype in DTYPE_MAP, f"dtype must be one of {list(DTYPE_MAP.keys())}"

    # Read base path from adapter_config.json
    peft_cfg = PeftConfig.from_pretrained(args.adapter_model_name_or_path)
    base_id = args.base_model_name_or_path or getattr(
        peft_cfg, "base_model_name_or_path", None
    )
    assert base_id, (
        "adapter_config.json does not include base_model_name_or_path; "
        "pass --base_model_name_or_path to override."
    )

    # Load base model and tokenizer
    model = AutoModel.from_pretrained(
        base_id, return_dict=True, dtype=DTYPE_MAP[args.dtype]
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id)

    # Attach adapter, merge, and unload PEFT layers
    model = PeftModel.from_pretrained(model, args.adapter_model_name_or_path)
    model.eval()
    model = model.merge_and_unload()  # plain transformers model

    # Save locally
    model.save_pretrained(args.output_model_name_or_path)
    tokenizer.save_pretrained(args.output_model_name_or_path)

    print(f"✓ merged model saved to: {args.output_model_name_or_path}")


if __name__ == "__main__":
    main()
