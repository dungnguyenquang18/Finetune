# QWEN 3 0.6B Finetune: Instruction & Chain-of-Thought (CoT) Fine-tuning

This repository contains scripts and data for fine-tuning a causal language model (LoRA / adapter style) on English->Vietnamese translation tasks, including support for chain-of-thought (CoT) supervision.

## What this project does

- Preprocesses parallel English-Vietnamese data and (optionally) generates CoT annotations using a generative API.
- Fine-tunes a pretrained causal LM with LoRA adapters, with configurable maximum token length, batch size and gradient accumulation. Quantization is not enabled by default in the provided training scripts; see the Quantization & Offload section below for instructions if you want to enable 8-bit/4-bit training.
- Provides small evaluation scripts to run quick checks and compute losses or generate outputs.

## Repository layout

- `data/` - raw and preprocessed datasets. Contains `train.json`, `train_cot.json`, `valid.json`, `valid_cot.json`, `test.json`, `test_cot.json` and preprocessing notebooks/scripts.
- `train/` - training scripts and model checkpoints. See `train.py` for the main Trainer pipeline.
- `evaluation/` - evaluation helpers and example evaluation scripts.

## Experimental Results

### Evaluation metrics (BLEU, chrF, BERTScore)

| Model              | Description                                   | BLEU-2 | BLEU-4 | chrF  | BERTScore F1 |
| ------------------ | --------------------------------------------- | ------ | ------ | ----- | ------------ |
| Base Model         | Pretrained base model (no fine-tuning)        | 6.53   | 2.64   | 13.08 | 0.644        |
| Normal Fine-tuning | Normal fine-tuning without chain-of-thought   | 28.00  | 20.74  | 40.18 | 0.724        |
| CoT Fine-tuning    | Fine-tuning with chain-of-thought supervision | 22.95  | 15.26  | 36.59 | 0.755        |

All metrics computed on n = 66 test samples.

### Analysis

- The base model without fine-tuning shows low performance with BLEU-2 = 6.53 and BLEU-4 = 2.64 (n = 66).
- Normal fine-tuning significantly improves performance vs. base:
  - BLEU-2 increased to 28.00 (+21.47 points)
  - BLEU-4 increased to 20.74 (+18.10 points)
- Chain-of-thought fine-tuning is lower than normal fine-tuning on BLEU but still clearly above base:
  - BLEU-2: 22.95 (vs. 28.00 for normal fine-tuning)
  - BLEU-4: 15.26 (vs. 20.74 for normal fine-tuning)
  - Note: On other metrics from the evaluation JSONs, CoT achieved BERTScore F1 ≈ 0.755 (higher than normal ≈ 0.724), while chrF was lower (≈ 36.59 vs ≈ 40.18 for normal).


## Folder structure

```
finetune/
├─ data/                          # Datasets + notebooks for preprocessing
│  ├─ train.json                  # EN→VI pairs (train)
│  ├─ train_cot.json              # EN→VI with CoT annotations (train)
│  ├─ valid.json                  # EN→VI pairs (validation)
│  ├─ valid_cot.json              # EN→VI with CoT (validation)
│  ├─ test.json                   # EN→VI pairs (test)
│  ├─ test_cot.json               # EN→VI with CoT (test)
│  ├─ crawl.ipynb                 # Data crawling notebook (optional)
│  └─ preprocess.ipynb            # Preprocessing / CoT generation (optional)
│
├─ evaluation/                    # Evaluation scripts + sample outputs
│  ├─ test_base.py                # Evaluate base model
│  ├─ test_normal_finetune.py     # Evaluate LoRA (no CoT)
│  ├─ test_cot_finetune.py        # Evaluate LoRA (with CoT)
│  ├─ test_base.json              # Example results (base)
│  ├─ test_normal_finetune.json   # Example results (normal finetune)
│  └─ test_cot_finetune.json      # Example results (CoT finetune)
│
├─ finetune/                      # Training scripts and produced adapters
│  ├─ finetune.py                 # LoRA training (no CoT)
│  ├─ finetuneCoT.py              # LoRA training (with CoT)
│  ├─ test.py                     # Small ad-hoc tests
│  ├─ qwen3_envi_lora/            # Saved adapter + tokenizer (normal)
│  │  ├─ adapter_model.safetensors
│  │  ├─ adapter_config.json
│  │  ├─ tokenizer.json | vocab.json | merges.txt | ...
│  │  └─ checkpoint-*/            # Training checkpoints
│  └─ qwen3_envi_lora_cot/        # Saved adapter + tokenizer (CoT)
│     ├─ adapter_model.safetensors
│     ├─ adapter_config.json
│     ├─ tokenizer.json | vocab.json | merges.txt | ...
│     └─ checkpoint-*/            # Training checkpoints
│
├─ README.md
└─ LICENSE
```

Notes:

- Paths in training scripts reference absolute Windows paths (e.g., `D:/Finetune/...`). Adjust as needed.
- Adapter directories (`qwen3_envi_lora*`) are Trainer outputs; you can point evaluation scripts to these.

## Key concepts and recommendations

- MAX_LEN is the number of tokens (not characters). Use `tokenizer.model_max_length` to check the model's context limit.
- If your training targets include chain-of-thought (CoT), keep the CoT text in the label (do not mask it) and ensure MAX_LEN is large enough to contain the full prompt + CoT.
- CoT increases memory usage — reduce batch size, enable gradient accumulation, enable gradient checkpointing, or use quantization/offloading if you hit OOM.
- LoRA/PEFT training updates only adapter parameters and is much lighter than full fine-tuning.

## Preprocessing

The `data/preprocess.py` script can call a generative API to produce CoT annotations. Typical flow:

1. Load `train.json` (list of {"en": ..., "vi": ...}).
2. For each example, send a prompt to the generative model to request a CoT-formatted explanation and the final Vietnamese sentence.
3. Save augmented examples to `train_cot.json`.

Notes:

- The example preprocessing script includes retry logic and saves progress after each example. If you plan to run long API jobs, keep incremental saves to avoid losing progress.
- Always respect API rate limits and rotate API keys if available.

## Training (high level)

Open `train/train.py` to configure training parameters. Important flags to check:

- `MODEL_ID` / pretrained model identifier
- `MAX_LEN` — maximum total tokens in input/labels (prompt + CoT + answer)
- `BATCH_SIZE` and `gradient_accumulation_steps`
- `num_train_epochs`, `learning_rate`
- `fp16` (this repo's example uses fp16) — note `load_in_8bit` is not set in the example scripts

Recommendations when using CoT:

- Increase `MAX_LEN` to accommodate CoT (e.g. 1024 or higher), but ensure your GPU can handle it.
- If memory is tight: set `per_device_train_batch_size=1`, increase `gradient_accumulation_steps` and enable `gradient_checkpointing`.
- Use `model.config.use_cache = False` to reduce memory usage during training with long contexts.
- If using quantization (8-bit / 4-bit) via bitsandbytes, ensure your GPU and environment are compatible. Older GPUs (e.g. P100) may not work with bnb quantization.

Example training-related checks before starting:

1. Inspect model and tokenizer maximum length:

```python
print('tokenizer.model_max_length =', tokenizer.model_max_length)
MAX_LEN = min(desired_max_len, tokenizer.model_max_length)
```

2. Quick forward sanity test to detect OOM early:

```python
tok = tokenizer('Hello ' + 'x ' * (MAX_LEN - 10), truncation=True, max_length=MAX_LEN, return_tensors='pt').to('cuda')
model.config.use_cache = False
with torch.no_grad():
    _ = model(**tok)
print('forward ok')
```

## Quantization & Offload

Quantization (8-bit / 4-bit) and offload can help when VRAM is limited, but they are not enabled by default in this repository's training scripts. The provided `train/train.py` uses fp16 + LoRA which is already memory-friendly.

If you want to enable quantized training with bitsandbytes, typical steps are:

- Install bitsandbytes and accelerate:

  - pip install bitsandbytes accelerate

- Load the model with quantization flags, for example:

  - AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_8bit=True, device_map='auto', trust_remote_code=True)

Or use the 4-bit options from bnb for more aggressive quantization following the Hugging Face recommendations.

Also consider these alternatives to reduce memory without quantization:

- Gradient checkpointing and `model.config.use_cache = False`.
- Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`.
- Use CPU/GPU offload with `accelerate` or DeepSpeed (ZeRO).

Caveats: bitsandbytes requires recent CUDA, driver and GPU architectures. Some older GPUs (for example many P100 setups) do not support 8-bit/4-bit bnb paths. Always test `import bitsandbytes as bnb` and a small forward pass before relying on quantization in a long training job.

## Evaluation

- The trainer uses `predict_with_generate=True` in some configurations to evaluate generated CoT and final answers.
- Choose evaluation metrics that make sense: token-level loss, BLEU or chrF for translation, or string equality if you target exact-match final answers.

## Running (PowerShell examples)

Install dependencies (example):

```powershell
python -m pip install -r requirements.txt
```

Run training (example):

```powershell
python train/train.py --config config.yml
```

Run preprocessing to generate CoT annotations (careful with API usage):

```powershell
python data/preprocess.py
```

## Troubleshooting

- Out of memory (OOM): reduce `MAX_LEN`, reduce batch, enable gradient accumulation, or enable checkpointing. Consider using a machine with more VRAM.
- bitsandbytes import fails: check CUDA, driver and GPU compute capability. If unsupported, do not use `load_in_8bit`.
- Slow training: lower precision (fp16), use LoRA, or use more hardware.





Updated: September 30, 2025

# Finetune
