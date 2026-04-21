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

### KV Cache (`cache_multihead_attention.py`)

| Component | Description |
|---|---|
| `CachedMultiHeadAttention` | MHA with KV cache; prefill stores full K/V, each decode step appends one token's K/V and attends over growing cache |
| `clear_cache` | Resets stored K/V tensors between sequences |

### Fine-tuning (`finetuning_lora.py`)

| Component | Description |
|---|---|
| `LoRALinear` | Frozen base weight `W` + low-rank adapters `A` and `B`; only `A`, `B` are trained |
| `QLoRALinear` | Same as LoRA but `W` stored in `float16`, adapters in `bfloat16` |
| `LoRAMultiHeadAttention` | MHA with LoRA applied to `W_q` and `W_v`; `W_k` and `W_o` remain frozen |

### Quantization (`quantization_scaling.py`)

| Component | Description |
|---|---|
| `quantize` | Maps float32 weights to int8 using min/max scaling; returns quantized tensor and scale factor |
| `dequantize` | Recovers approximate float values by multiplying int8 weights by the scale factor |
| `quantization_error` | Measures mean absolute error between original and dequantized weights |

### Speculative Decoding (`speculative_decoding.py`)

| Component | Description |
|---|---|
| `get_next_token_probs` | Runs a model logits function and returns a softmax probability distribution over vocab |
| `sample_token` | Samples a single token from a probability distribution via multinomial sampling |
| `speculative_decode` | Drafts `n_draft` tokens with a small model, verifies with the large target model using rejection sampling; guarantees output distribution matches the target model exactly |

---

## Files

```
transformer_components.py       # Full model in PyTorch — MiniGPT + all building blocks
train.py                        # Training script — cross-entropy loss, Adam optimizer
finetuning_lora.py              # LoRA and QLoRA implementations with shape/gradient tests
cache_multihead_attention.py    # MHA with KV cache — prefill + autoregressive decode demo
quantization_scaling.py         # Int8 quantization with min/max scaling — 7 tests
speculative_decoding.py         # Speculative decoding with rejection sampling — 5 tests
transformer_components.rs       # Inference-only Rust impl using ndarray
transformer_components.ts       # Inference-only TypeScript impl (zero dependencies)
practice_sample.ipynb           # Notebook for experimentation
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

### Python — KV Cache

```bash
python cache_multihead_attention.py   # prefill + 3 decode steps + memory estimate
```

### Python — Quantization

```bash
python quantization_scaling.py   # 7 tests: dtype, range, scale, error, memory, dequant, layer sim
```

### Python — Speculative Decoding

```bash
python speculative_decoding.py   # 5 tests: output length, acceptance rate, disagreement, growth, free token
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

### KV Cache

During autoregressive generation, the model produces one token at a time. Without caching, every new token recomputes K and V for all prior positions — O(n) work per step, O(n²) total.

KV caching stores K and V after the **prefill** (processing the full prompt) and appends only the new token's K/V at each **decode** step:

```
Prefill (use_cache=False):  x(batch, seq, d)  → K/V cached, full attention over seq tokens
Decode  (use_cache=True):   x(batch,   1, d)  → new K/V appended, attention over (seq+t) tokens
```

Memory cost scales as `2 × layers × batch × heads × (seq + decode_steps) × d_k × dtype_bytes`. At `seq=10, 3 decode steps, 32 layers, float16` the cache is ~0.05 MB — grows linearly with context length.

### Quantization

INT8 quantization maps each float32 weight to an 8-bit integer using a linear scale derived from the weight's min and max:

```
scale          = (max(W) - min(W)) / 255
W_quantized    = round(W / scale).clamp(-128, 127)   # int8
W_dequantized  = W_quantized * scale                  # approx float32
```

This gives a **4× memory reduction** (float32 → int8) with a small, bounded quantization error (typically < 0.02 mean absolute error on normally distributed weights). The scale factor must be stored alongside the quantized weights for dequantization.

### Speculative Decoding

Speculative decoding speeds up large model inference without any quality loss. A small draft model proposes `n_draft` tokens cheaply; the large target model verifies all of them in a single forward pass using rejection sampling:

```
Accept draft token x  if:  rand(0, 1) < min(1, p_target(x) / p_draft(x))
```

- `p_target ≥ p_draft` → always accept (target model agrees or is more confident)
- `p_target < p_draft` → accept with probability = ratio (draft was overconfident)

If all `n_draft` tokens are accepted, a bonus token is sampled from the target model for free. This guarantees the output distribution exactly matches the large model (**lossless**). Practical speedup is ~3–4× on predictable text with a well-matched draft model.

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

### cache_multihead_attention.py

| Param | Value |
|---|---|
| `d_model` | 256 |
| `heads` | 4 |
| `batch` | 1 |
| `seq` (prefill) | 10 |
| decode steps | 3 |
| layers (memory est.) | 32 |

### quantization_scaling.py

| Param | Value |
|---|---|
| `n_bits` | 8 |
| test weight shape | up to 1024×1024 |
| simulated `d_model` | 512 |
| simulated layers | W_q, W_k, W_v, W_o |

### speculative_decoding.py

| Param | Value |
|---|---|
| `vocab_size` | 100 |
| `n_draft` | 4 |
| `max_new_tokens` | 8–20 |
| `temperature` | 1.0 |
# miniGPT
