# Transformer from Scratch

A hands-on implementation of a GPT-style transformer and LoRA/QLoRA fine-tuning — built from the ground up in **Python** (with PyTorch), **Rust**, and **TypeScript**.

The goal is to understand the internals by writing every component by hand — no high-level abstractions, just the math.

---

## What's implemented

### Core transformer (`transformer_components.py`, `.rs`, `.ts`)

| Component | Description |
|---|---|
| `scaled_dot_product_attention` | Scores, causal masking, softmax, weighted sum |
| `MultiHeadAttention` | Splits into `h` heads, attends in parallel, concatenates and projects |
| `LayerNorm` | Normalizes each token's embedding to zero mean, unit variance |
| `FeedForward` | Two-layer MLP with ReLU: `d_model → 4×d_model → d_model` |
| `TransformerBlock` | MHA + residual + LayerNorm + FFN + residual |
| `PositionalEncoding` | Sinusoidal position encoding added to token embeddings |
| `MiniGPT` | Full decoder-only model: embedding → PE → N blocks → logit projection |

### Fine-tuning (`finetuning_lora.py`)

| Component | Description |
|---|---|
| `LoRALinear` | Frozen base weight `W` + low-rank adapters `A` and `B`; only `A`, `B` are trained |
| `QLoRALinear` | Same as LoRA but `W` stored in `float16`, adapters in `bfloat16` |
| `LoRAMultiHeadAttention` | MHA with LoRA applied to `W_q` and `W_v`; `W_k` and `W_o` remain frozen |

---

## Files

```
transformer_components.py   # Full model in PyTorch — MiniGPT + all building blocks
train.py                    # Training script — cross-entropy loss, Adam optimizer
finetuning_lora.py          # LoRA and QLoRA implementations with shape/gradient tests
transformer_components.rs   # Inference-only Rust impl using ndarray
transformer_components.ts   # Inference-only TypeScript impl (zero dependencies)
practice_sample.ipynb       # Notebook for experimentation
```

---

## Quickstart

### Python — core model

```bash
pip install torch
python transformer_components.py   # shape checks
python train.py                     # 10-epoch training run
```

### Python — LoRA fine-tuning

```bash
python finetuning_lora.py   # runs 7 tests: shapes, zero-init, grad flow, rank sweep
```

### Rust

```bash
cargo add ndarray
cargo run
```

### TypeScript

```bash
npx ts-node transformer_components.ts
```

---

## Architecture

### MiniGPT (decoder-only, GPT-style)

```
tokens (batch, seq)
  ↓  Embedding          → (batch, seq, d_model)
  ↓  PositionalEncoding → (batch, seq, d_model)
  ↓  TransformerBlock × N
  ↓  Linear projection  → (batch, seq, vocab_size)
  ↓  logits
```

A causal mask (lower-triangular) inside every attention layer ensures each token can only attend to earlier positions — this is what makes it a decoder.

Attention is **O(n²)** in sequence length: a sequence of length `n` produces an `n×n` weight matrix per head per layer. This is the core bottleneck that motivated FlashAttention, Longformer, etc.

### LoRA

Standard fine-tuning updates all weights. LoRA freezes the pretrained weight `W` and injects a low-rank update:

```
output = W(x) + x @ A @ B
```

`W` is `(d_model, d_model)`. `A` is `(d_model, rank)` and `B` is `(rank, d_model)` — only these two matrices are trained. At `rank=8` and `d_model=512`, the reduction is ~32x fewer trainable parameters than full fine-tuning.

`B` is initialized to zeros, so LoRA adds nothing at the start — training begins from the pretrained model's behavior.

**Why `W_q` and `W_v` only?** `W_q` controls what patterns a token searches for (task-specific). `W_v` controls what gets written to the output (task-specific). `W_k` controls how a token presents itself — relatively stable across tasks.

**QLoRA** stores `W` in `float16` (4× memory saving over full precision) while keeping the adapters in `bfloat16`. Quantization error in the frozen weights is acceptable since only the adapters update during training.

---

## Default hyperparameters

### train.py

| Param | Value |
|---|---|
| `vocab_size` | 256 |
| `d_model` | 128 |
| `heads` | 4 |
| `num_layers` | 2 |
| `batch` | 4 |
| `seq` | 32 |
| `lr` | 1e-3 |
| `epochs` | 10 |

### finetuning_lora.py

| Param | Value |
|---|---|
| `d_model` | 256 / 512 |
| `rank` | 8 |
| `heads` | 8 |
| `batch` | 2 |
| `seq` | 10 |
