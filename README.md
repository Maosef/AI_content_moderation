# TLDD + SFT

A quickstart for running TLDD, specifically with Alpaca, but it contains the code to run it without. It covers:

* building TLDD (with BackdoorLLM)
* building the TLDD-processed Alpaca dataset,
* training baseline vs. TLDD adapters,
* and evaluating with `lm_eval`.


## 0) Prerequisites

* Python 3.9+ (GPU recommended for speed; CPU works but is slow).
* (Optional) CUDA-capable GPU and drivers.
* Hugging Face model access for the chosen models.

## 1) Environment & Dependencies

Create/activate an environment (conda/venv), then:

```bash
pip install --upgrade pip
# Install PyTorch first per your platform (CPU or CUDA) from pytorch.org, then:
pip install transformers datasets trl peft accelerate sentence-transformers openai tqdm numpy
# For the lm-eval CLI:
pip install lm-eval
```

If you’re using the **Golden Retriever** bits, ensure the `golden/` module is importable (it lives in this repo). The simplest way is to run commands from the project root so `golden/` is on `PYTHONPATH`.

## 2) Configure Environment Variables

Recommended:

```bash
# Optional: put the HF cache on a big/fast disk
export HF_HOME=~/hf_cache

# Use OpenAI’s API (preferred for TLDD rewriting) …
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o"          # optional override

# …or use a local OpenAI-compatible endpoint (default in code is http://localhost:11434/v1).
# Make sure your local server is running and has the model ID you pass via --model.
```

Quality-of-life (optional):

```bash
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TORCHVISION=1   # avoids loading torchvision for text-only
```

## 3) Build the TLDD-processed Alpaca dataset

Single-machine command (GPU):

```bash
python tldd.py \
  --make-tldd-alpaca \
  --use-golden-rag \
  --rag-top-k 3 \
  --rag-model-id sentence-transformers/all-MiniLM-L6-v2 \
  --rag-device cuda \
  --rag-max-seq-length 512 \
  --alpaca-split train \
  --save-path ./tldd_alpaca.hf \
  --use-openai        # drop this flag to use your local OpenAI-compatible endpoint
```

Tips:

* No GPU? switch `--rag-device cpu`.
* Faster preprocessing on one box: add `--map-num-proc 8`.
* Using a local model server? remove `--use-openai` and (optionally) set `--model <local-model-id>`.

### (Optional) Sharded build locally to make it faster

```bash
N=16
for i in $(seq 0 $((N-1))); do
  python tldd.py \
    --make-tldd-alpaca --use-golden-rag --use-openai \
    --rag-model-id sentence-transformers/all-MiniLM-L6-v2 \
    --rag-device cuda --alpaca-split train \
    --num-shards "$N" --shard-id "$i" \
    --save-path "./shards/tldd_alpaca.shard${i}-of-${N}.hf"
done
```

Then merge shards:

```python
# save as merge_shards.py and run: python merge_shards.py
import os, glob
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets

shard_dirs = sorted(glob.glob("shards/tldd_alpaca.shard*-of-*.hf"))
splits = {}
for sd in shard_dirs:
    ds = load_from_disk(sd)
    if isinstance(ds, DatasetDict):
        for k, v in ds.items():
            splits.setdefault(k, []).append(v)
    else:
        splits.setdefault("train", []).append(ds)

merged = {k: concatenate_datasets(vs) for k, vs in splits.items()}
DatasetDict(merged).save_to_disk("tldd_alpaca.hf")
print("Merged to ./tldd_alpaca.hf")
```

## 4) Train a **baseline** adapter on Alpaca

Small model example (single GPU):

```bash
python openhermes_forNoopur.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --output-dir runs_baseline/Llama-3.2-1B-Instruct-alpaca \
  --batch-size 8 \
  --lora-r 16 --lora-alpha 16 \
  --learning-rate 5e-5 \
  --quantization none
```

## 5) Train with **TLDD** data

```bash
python openhermes_forNoopur.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --tldd-path ./tldd_alpaca.hf \
  --output-dir runs_tldd/Llama-3.2-1B-Instruct-tldd \
  --batch-size 8 \
  --lora-r 16 --lora-alpha 16 \
  --learning-rate 5e-5 \
  --quantization none
```

## 6) Evaluate (lm-eval-harness)

Baseline:

```bash
lm_eval \
  --model hf \
  --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,peft=runs_baseline/Llama-3.2-1B-Instruct-alpaca" \
  --tasks "hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande,triviaqa" \
  --batch_size auto \
  --device cuda:0 \
  --output_path eval_results/Llama-3.2-1B-Instruct_baseline
```

TLDD:

```bash
lm_eval \
  --model hf \
  --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct,peft=runs_tldd/Llama-3.2-1B-Instruct-tldd" \
  --tasks "hellaswag,lambada_openai,piqa,arc_easy,arc_challenge,winogrande,triviaqa" \
  --batch_size auto \
  --device cuda:0 \
  --output_path eval_results/Llama-3.2-1B-Instruct_tldd
```

## What you’ll see on disk

* `tldd_alpaca.hf/` — TLDD-processed Alpaca dataset (`datasets.save_to_disk` format).
* `runs_baseline/<MODEL>-alpaca/` — baseline LoRA adapter + tokenizer.
* `runs_tldd/<MODEL>-tldd/` — TLDD LoRA adapter + tokenizer.
* `eval_results/` — JSON and metrics from `lm_eval`.
* `logs_*` — trainer logs created per run.

## Common flags (quick reference)

**`tldd.py`**

* `--make-tldd-alpaca` — build dataset and exit.
* `--use-openai` / `--model` — choose cloud OpenAI vs. local OpenAI-compatible model.
* RAG: `--use-golden-rag --rag-top-k 3 --rag-model-id sentence-transformers/all-MiniLM-L6-v2 --rag-device {cuda|cpu}`.
* Throughput: `--map-num-proc <N>`.
* Sharding: `--num-shards N --shard-id I`.

**`openhermes_forNoopur.py`**

* Core: `--model-id`, `--output-dir`, `--batch-size`, `--learning-rate`.
* LoRA: `--lora-r`, `--lora-alpha`.
* Dataset: `--tldd-path ./tldd_alpaca.hf` (if using TLDD), otherwise defaults to raw Alpaca.
* Quantization: `--quantization {none|bnb4}` (4-bit requires bitsandbytes support).

## Compatible with BackdoorLLM

This script also works with running just TLDD with the BackdoorLLM repository.

To do that, complete the instructions within their repository in the attack/DPA folder.

Then take the path of the original jailbreaks and the models (baseline and finetuned) and pass through TLDD to get the new jailbreaks. 