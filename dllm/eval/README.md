# Evaluation

We provide a **unified evaluation framework** built on top of **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**, serving as the standardized backbone for evaluating the [LLaDA series,](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) [Dream series](https://huggingface.co/collections/Dream-org/dream-7b), and BERT-diffusion models.
It supports diverse model architectures and evaluation paradigms through a **configuration-driven**, **modular**, and **extensible** design.



## Table of Contents
1. [Setup](#setup)
   - [Environment Variables](#environment-variables)
   - [Dependencies](#dependencies)
2. [Evaluation](#evaluation)
   - [Run Command](#run-command)
   - [Plausible Tasks](#plausible-tasks)
   - [Example Evaluation Results](#example-evaluation-results)
3. [Framework and Further Extension](#framework-and-further-extension)
   - [File Structure](#file-structure)


---

## Setup

> [!IMPORTANT]
> Before running evaluations, you **must** export the required environment variables to specify dataset and model paths.  
> These paths tell the evaluation framework where to locate model checkpoints and datasets, and where to cache evaluation results for **lm-eval**.  

### Environment Variables

Before running evaluations, export the following environment variables to specify where datasets, pretrained models, and caches are stored:

```bash
export BASE_DATASETS_DIR=<path_to_huggingface_datasets>
export BASE_MODELS_DIR=<path_to_local_or_shared_models>
export HF_DATASETS_CACHE=<path_to_hf_dataset_cache>
export HF_EVALUATE_CACHE=<path_to_hf_evaluate_cache>
export PYTHONPATH=.:$PYTHONPATH
```


### Dependencies

Install the core dependencies:

```bash
pip install -e lm-evaluation-harness
pip install accelerate transformers datasets
pip install -e ".[ifeval,math]"
```

Make sure to initialize submodules before installation:
```bash
git submodule update --init --recursive
```


## Evaluation

### Run Command

> [!NOTE]
> All configuration parameters (few-shot, max length, temperature, etc.) are aligned with model's original repo.
> You can now directly execute task-specific shell scripts under `scripts/` for one-line evaluation.

**Example commands:**

```bash
bash scripts/eval_dream_instruct.sh
bash scripts/eval_dream_base.sh
bash scripts/eval_llada_base.sh
bash scripts/eval_llada_instruct.sh
bash scripts/eval_bert_base.sh
```

Each script loads its corresponding configurations and launches evaluation automatically.
You no longer need to manually specify `model_class`, `task_name`, or `model_path` arguments.

### Plausible Tasks

| Category | Tasks |
|----------|-------|
| **Instruct** | `mmlu_generative`, `mmlu_pro`, `gsm8k_cot`, `minerva_math`, `gpqa_main_n_shot`, `humaneval_instruct`, `mbpp_instruct`, `ifeval` |
| **Base** | `humaneval`, `gsm8k_cot`, `mbpp`, `minerva_math`, `bbh`, `mmlu`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `gpqa_main_n_shot`, `winogrande`, `race` |

> [!NOTE]
> Certain dataset configurations of lm-eval were refined to match the model templates and ensure optimal evaluation performance.


### Example Evaluation Results



<details>
<summary><strong>LLaDA-Base results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



<details>
<summary><strong>LLaDA-Instruct results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



<details>
<summary><strong>Dream-Base results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>


<details>
<summary><strong>Dream-Instruct results</strong></summary>

| Source | BBH | GSM8K | Math | HumanEval | MBPP |
|--------|-----|-------|------|-----------|------|
| **Reported** | — | — | — | — | — |
| **Reproduced** | — | — | — | — | — |

</details>



## Framework and Further Extension

> [!NOTE]
> Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements model-specific generation and likelihood computation methods.

Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements:

- **`generate_until()`** — defines model-specific text generation (e.g., diffusion or autoregressive).
- **`loglikelihood()`** — computes NLL or masked likelihood (Monte Carlo, autoregressive, etc.).
- **`apply_chat_template()`** — formats multi-turn inputs when `--apply_chat_template=True`.

This modular design allows adding new model architectures while keeping the evaluation pipeline unified.

### Customizing Tasks

> [!NOTE]
> Customize evaluation behavior by editing YAML configuration files — no code changes required.

To customize or extend tasks, edit the configuration files in:

```
lm-evaluation-harness/lm_eval/tasks/<task_name>/<task_name>.yaml
```

For example:

```
lm-evaluation-harness/lm_eval/tasks/mbpp/mbpp.yaml
```

Each YAML file defines:

- Dataset sources and splits
- Prompt templates and context formatting
- Metric computation and postprocessing
- Stop sequences and answer extraction rules

By editing these YAMLs, you can modify task behavior or introduce new benchmarks without rebuilding the framework.

### Adding New Models

> [!NOTE]
> New model types can be integrated while maintaining **full compatibility** with the unified evaluation system.

To integrate a new model type:

1. **Create a new evaluation file**, e.g. `dllm/eval/eval_newmodel.py`.

2. **Register it with lm-eval**:

   ```python
   from lm_eval.api.registry import register_model

   @register_model("newmodel")
   class NewModel(LM):
       ...
   ```

3. **Implement `generate_until()` and `loglikelihood()`** for the model's decoding logic.

4. **Add corresponding entries to `eval_configs.sh`** for task configurations.

> [!NOTE]
> This approach supports both custom and standard model backends, making the framework highly extensible.

## Acknowledgments
We sincerely thank [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their outstanding contributions to the open evaluation ecosystem