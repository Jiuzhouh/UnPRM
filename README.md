# UnPRM
The official code for AAAI 2026 paper [Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning](https://arxiv.org/pdf/2508.01773).

## Repository Layout

- `train_prm_main.py`: Train a PRM (a causal LM + value head) with LoRA fine-tuning.
- `generate_train_data.py`: Generate candidate solutions with vLLM and label steps via search (binary / sequential / uncertainty-driven).
- `prm_verifier.py`: Example pipeline: sample solutions from a math LLM and score each step with a PRM.
- `inference_with_prm.py`: More PRM scoring variants (different separator tokens / PRM checkpoints).
- `value_model.py`: `AutoModelForCausalLMWithValueHead` wrapper (adapted from TRL).
- `utils.py`: Prompt templates, answer extraction, verifier helpers, uncertainty metrics.
- `math_utils.py`: Math answer normalization/equivalence helpers used by data generation.
- `data/un_prm_data_40k.json`: Step-labeled training examples (see schema below).

## Data Schemas

### JSON step-labeled format (used by `data/*.json`)

Each file is a JSON list of items:

```json
{
  "prompt": "...question...",
  "answer": "...final answer...",
  "completions": ["Step 1: ...", "Step 2: ...", "..."],
  "label": [true, true, false]
}
```

`label[i]` is the correctness label for `completions[i]`.

## How PRM Supervision Is Represented

`train_prm_main.py` inserts a special **step separator token** between steps. The default is the single character `ş`.

During training, the script:

- builds a single input sequence: `{query}\n{answer}`
- formats `answer` as: `Step 1 ... ş\nStep 2 ... ş\n... ş`
- gathers the model value predictions at the `ş` token positions
- applies one of the supported losses: `rank`, `orm`, `mse`, `bce`

## Setup

This repo is GPU-oriented (Transformers + optional FlashAttention + optional 8-bit loading + optional vLLM).

Suggested environment:

- Python 3.10+
- CUDA-enabled PyTorch (if training/scoring on GPU)

Typical dependencies:

```bash
pip install -U \
  torch transformers accelerate datasets peft trl \
  bitsandbytes \
  pandas numpy tqdm wandb \
  sympy pydantic \
  vllm
```

Notes:

- Some model loads set `trust_remote_code=True`.
- `train_prm_main.py` enables `attn_implementation="flash_attention_2"`; if your environment/GPU doesn't support it, remove that argument.

## Generate And Label PRM Training Data

Commands used in this repo:

```bash
# Generate solutions
python generate_train_data.py \
  --seed 22 \
  --model_dir Qwen/Qwen2.5-7B-Instruct \
  --data_dir data/sampled_math_questions.json \
  --save_dir qwen_outputs.jsonl \
  --start 0 \
  --generate

# Label steps (uncertainty-delta-driven)
python generate_train_data.py \
  --seed 22 \
  --model_dir Qwen/Qwen2.5-7B-Instruct \
  --synthetic_data_dir data/qwen_outputs_filtered.jsonl \
  --save_dir qwen_outputs_labelled_uncertainty_delta_driven_entropy.jsonl \
  --start 103 \
  --get_label \
  --use_uncertainty_delta_driven \
  --use_uncertainty_delta_driven
```

Label/search params:

```text
--use_adaptive_binary
--sequential_search
--use_uncertainty_driven
--use_uncertainty_delta_driven
--uncertainty_method entropy
--uncertainty_driven_data_generation
```

## Train a PRM

Example (adjust paths/model names for your machine):

```bash
python train_prm_main.py \
  --dataset_path data/un_prm_data_40k.json \
  --model_path Qwen/Qwen2.5-Math-7B-Instruct \
  --save_path qwen_unprm_checkpoints \
  --loss_type rank \
  --zeta 4
```

What you get:

- `--save_path/`: HuggingFace `save_pretrained(...)` output for the reward model and tokenizer.

## Score Steps With a PRM (Inference)

Two common scoring patterns appear in the example scripts:

1) **Separator-token scoring** (e.g., Qwen PRMs): build a conversation where steps are joined by a special token like `<extra_0>`, then read the model's probabilities at those token positions.

2) **Plus/minus classification scoring** (used in `prm_verifier.py` / `inference_with_prm.py`): for each step, append an assistant response of `+` and read the probability of `+` vs `-` from logits near the end of the prompt.

These scripts are intended as working examples you can adapt to your own data and checkpoints.
