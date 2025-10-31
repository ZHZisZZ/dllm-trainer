"""
Interactive chat / generation script for LLaDA models.

Examples
--------
# Chat mode (multi-turn, chat template)
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH" --chat True

# Raw single-turn generation
python -u examples/llada/chat.py --model_name_or_path "YOUR_MODEL_PATH" --chat False
"""
import sys
from dataclasses import dataclass
import transformers

import dllm
from dllm.pipelines import llada
from dllm.tools.chat import multi_turn_chat, single_turn_generate



@dataclass
class ScriptArguments:
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    seed: int = 42
    chat: bool = True
    visualize: bool = True

    def __post_init__(self):
        # same base-path resolution logic as in generate.py
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class GeneratorConfig(llada.LLaDAGeneratorConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


def main():
    parser = transformers.HfArgumentParser(
        (ScriptArguments, GeneratorConfig)
    )
    script_args, gen_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    generator = llada.LLaDAGenerator(model=model, tokenizer=tokenizer)

    if script_args.chat:
        multi_turn_chat(
            generator=generator,
            gen_config=gen_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn generation (no chat template).")
        single_turn_generate(
            generator=generator,
            gen_config=gen_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
