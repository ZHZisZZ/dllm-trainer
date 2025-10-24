# Unified Model Evaluation Framework

This repository provides a **unified evaluation interface** built upon **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** for assessing generative and likelihood-based language models.  
It supports multiple architectures and evaluation paradigms through a **configuration-driven** and **extensible** design.

---

## Table of Contents
1. [Setup](#setup)
   - [Environment Variables](#environment-variables)
   - [Dependencies](#dependencies)
2. [File Structure](#file-structure)
3. [Evaluation](#evaluation)
   - [Run Command](#run-command)
   - [Plausible Tasks](#plausible-tasks)
   - [Example Evaluation Results](#example-evaluation-results)
4. [Framework and Further Extension](#framework-and-further-extension)

---

## Setup

> [!IMPORTANT]
> Before running evaluations, you **must** export the required environment variables to specify dataset and model paths.
> These paths tell the evaluation framework where to locate model checkpoints and datasets, and where to cache evaluation results for lm-eval.

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


---

## File Structure

| File | Purpose |
|------|---------|
| `eval_model.sh` | Main launcher for model evaluation. Sets up distributed environment variables, loads task configs, and runs the evaluation script. |
| `eval_configs.sh` | Contains unified per-task configurations (few-shot, max length, temperature, etc.) for all model classes. |
| `dllm/eval/eval_*.py` | Defines model-specific evaluation logic extending `lm_eval.api.model.LM`. Each file handles generation, NLL computation, and task integration. |


---

## Evaluation

### Run Command

> [!IMPORTANT]
> All configuration parameters (few-shot, steps, temperature, etc.) are **automatically loaded** from `eval_configs.sh` — no manual configuration needed!

**Basic usage:**

```bash
bash eval_model.sh <model_class> <task_name> <model_path>
```

**Example:**

```bash
bash eval_model.sh dream gsm8k Dream-org/Dream-v0-Instruct-7B
```

**Arguments:**

| Argument | Description | Examples |
|----------|-------------|----------|
| `<model_class>` | Model type identifier | `dream`, `llada` |
| `<task_name>` | Evaluation benchmark | `gsm8k`, `mmlu`, `mbpp` |
| `<model_path>` | Model location | `Dream-org/Dream-v0-Instruct-7B` (HF) or `/path/to/model` (local) |

### Plausible Tasks

| Category | Tasks |
|----------|-------|
| **Instruct** | `mmlu_generative`, `mmlu_pro`, `gsm8k_cot`, `minerva_math`, `gpqa_main_n_shot`, `humaneval_instruct`, `mbpp_instruct`, `ifeval` |
| **Base** | `humaneval`, `gsm8k_cot`, `mbpp`, `minerva_math`, `bbh`, `mmlu`, `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `gpqa_main_n_shot`, `winogrande`, `race` |

> [!TIP]
> Choose **Instruct** tasks for instruction-tuned models and **Base** tasks for pretrained models. Model's task-specific parameter are stored within /eval/eval_configs.sh


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

---

## Framework and Further Extension

> [!NOTE]
> Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements model-specific generation and likelihood computation methods.

Each evaluation script in `dllm/eval/` subclasses `lm_eval.api.model.LM` and implements:

- **`generate_until()`** — defines model-specific text generation (e.g., diffusion or autoregressive).
- **`loglikelihood()`** — computes NLL or masked likelihood (Monte Carlo, autoregressive, etc.).
- **`apply_chat_template()`** — formats multi-turn inputs when `--apply_chat_template=True`.

This modular design allows adding new model architectures while keeping the evaluation pipeline unified.

### Customizing Tasks

> [!TIP]
> Customize evaluation behavior by editing YAML configuration files — no code changes required!

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

> [!IMPORTANT]
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

