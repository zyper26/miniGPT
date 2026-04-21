import torch

def quantize(W: torch.Tensor, n_bits=8):
    min_W, max_W = torch.min(W), torch.max(W)
    scaling_factor = (max_W - min_W)/255
    quantized_W = torch.round(W/scaling_factor).clamp(-128, 127).to(torch.int8)
    return quantized_W, scaling_factor

def dequantize(W_quant: torch.Tensor, scale: float):
    return torch.multiply(W_quant, scale)

def quantization_error(W_original, W_dequantized):
    return torch.abs(W_original - W_dequantized).mean()


torch.manual_seed(42)
 
 
# ── Test 1: Basic shape and dtype ─────────────────────────────
print("=" * 55)
print("Test 1: Shape and dtype")
print("=" * 55)
W       = torch.randn(4, 4)
W_q, s  = quantize(W)
W_deq   = dequantize(W_q, s)
print(f"Input dtype    : {W.dtype}")
print(f"Quantized dtype: {W_q.dtype}  (should be torch.int8)")
print(f"Dequant dtype  : {W_deq.dtype}  (should be torch.float32)")
print(f"Shape preserved: {W.shape == W_q.shape == W_deq.shape}")
print(f"PASS: {W_q.dtype == torch.int8 and W.shape == W_q.shape}")
 
 
# ── Test 2: Int8 range clamped ────────────────────────────────
print()
print("=" * 55)
print("Test 2: Values clamped to int8 range [-128, 127]")
print("=" * 55)
W_q, _ = quantize(torch.randn(100, 100) * 10)
min_val = W_q.min().item()
max_val = W_q.max().item()
print(f"Min quantized value: {min_val}  (should be >= -128)")
print(f"Max quantized value: {max_val}  (should be <= 127)")
print(f"PASS: {min_val >= -128 and max_val <= 127}")
 
 
# ── Test 3: Scale factor correct ──────────────────────────────
print()
print("=" * 55)
print("Test 3: Scale factor = (max - min) / 255")
print("=" * 55)
W          = torch.tensor([-2.4, 0.0, 1.5, 3.1])
W_q, scale = quantize(W)
expected   = (3.1 - (-2.4)) / 255
print(f"Scale computed : {scale:.6f}")
print(f"Scale expected : {expected:.6f}")
print(f"PASS: {abs(scale - expected) < 1e-5}")
 
 
# ── Test 4: Quantization error is small ───────────────────────
print()
print("=" * 55)
print("Test 4: Mean quantization error < 0.05")
print("=" * 55)
W       = torch.randn(512, 512)
W_q, s  = quantize(W)
W_deq   = dequantize(W_q, s)
error   = quantization_error(W, W_deq)
print(f"Mean abs error : {error:.6f}  (should be < 0.05)")
print(f"PASS: {error < 0.05}")
 
 
# ── Test 5: Memory reduction ──────────────────────────────────
print()
print("=" * 55)
print("Test 5: 4x memory reduction float32 → int8")
print("=" * 55)
W       = torch.randn(1024, 1024)
W_q, _  = quantize(W)
f32_mem = W.nelement() * 4
i8_mem  = W_q.nelement() * 1
print(f"float32 memory : {f32_mem:,} bytes ({f32_mem//1024//1024} MB)")
print(f"int8 memory    : {i8_mem:,} bytes ({i8_mem//1024//1024} MB)")
print(f"Reduction      : {f32_mem // i8_mem}x")
print(f"PASS: {f32_mem // i8_mem == 4}")
 
 
# ── Test 6: Dequantize recovers approximate values ────────────
print()
print("=" * 55)
print("Test 6: Dequantized values close to original")
print("=" * 55)
W       = torch.tensor([-2.4, -1.0, 0.0, 1.5, 3.1])
W_q, s  = quantize(W)
W_deq   = dequantize(W_q, s)
print("Original → Quantized → Dequantized:")
for orig, q, deq in zip(W.tolist(), W_q.tolist(), W_deq.tolist()):
    err = abs(orig - deq)
    print(f"  {orig:6.3f} → {q:5d} → {deq:6.3f}  (error: {err:.4f})")
max_err = torch.abs(W - W_deq).max().item()
print(f"Max error: {max_err:.4f}  (should be < 0.05)")
print(f"PASS: {max_err < 0.05}")
 
 
# ── Test 7: Larger model weight simulation ────────────────────
print()
print("=" * 55)
print("Test 7: Simulated transformer layer quantization")
print("=" * 55)
d_model = 512
layers  = {"W_q": torch.randn(d_model, d_model),
           "W_k": torch.randn(d_model, d_model),
           "W_v": torch.randn(d_model, d_model),
           "W_o": torch.randn(d_model, d_model)}
 
total_f32 = 0
total_i8  = 0
errors    = []
 
for name, W in layers.items():
    W_q, s  = quantize(W)
    W_deq   = dequantize(W_q, s)
    err     = quantization_error(W, W_deq)
    errors.append(err)
    total_f32 += W.nelement() * 4
    total_i8  += W_q.nelement() * 1
    print(f"  {name}: error={err:.5f}")
 
print(f"\nTotal float32 : {total_f32/1e6:.2f} MB")
print(f"Total int8    : {total_i8/1e6:.2f} MB")
print(f"Reduction     : {total_f32//total_i8}x")
print(f"Avg error     : {sum(errors)/len(errors):.5f}")
print(f"PASS: {total_f32//total_i8 == 4 and sum(errors)/len(errors) < 0.05}")
 
 
print()
print("=" * 55)
print("All tests complete")
print("=" * 55)