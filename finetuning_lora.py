import torch
import torch.nn.functional as F
import torch.nn as nn
from transformer_components import *

'''
QLoRA quantizes the frozen base weights to 4-bit NF4, 
reducing memory 4x over LoRA, while keeping adapters in bf16. 
The key insight is that quantization error in the frozen weights 
is acceptable since we're not updating them — only the low-rank adapters 
matter for task adaptation.
'''

class LoRALinear(nn.Module):
    def __init__(self, d_model, rank=8):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.W.weight.requires_grad = False

        self.A = nn.Parameter(torch.randn(d_model, rank))
        self.B = nn.Parameter(torch.zeros(rank, d_model))

    def forward(self, x):
        # output = W(x) + B·A·x
        return self.W(x) + x@self.A@self.B

class QLoRALinear(nn.Module):
    def __init__(self, d_model, rank=8):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.W = self.W.to(torch.float16)        
        self.W.weight.requires_grad = False     

        self.A = nn.Parameter(torch.randn(d_model, rank, dtype=torch.bfloat16))
        self.B = nn.Parameter(torch.zeros(rank, d_model, dtype=torch.bfloat16))

    def forward(self, x):
        w_out = self.W(x.to(torch.float16)).to(torch.bfloat16)
        return w_out + x.to(torch.bfloat16) @ self.A @ self.B

class LoRAMultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, rank=8):
        super().__init__()
        self.heads = heads
        assert d_model%heads == 0
        self.d_model = d_model

        self.W_q = LoRALinear(d_model=d_model, rank=rank)
        self.W_v = LoRALinear(d_model=d_model, rank=rank)

        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_k.weight.requires_grad=False

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.W_o.weight.requires_grad=False

    def forward(self, Q, K, V, mask=None):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        batch = Q.size(0)
        seq_len = Q.size(1)

        Q, K, V = Q.view(batch, seq_len, self.heads, self.d_model//self.heads).transpose(1, 2), \
                K.view(batch, seq_len, self.heads, self.d_model//self.heads).transpose(1, 2), \
                V.view(batch, seq_len, self.heads, self.d_model//self.heads).transpose(1, 2), \

        output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(output), weights

'''
W_q controls: what patterns does this token search for?    ← task-specific
W_v controls: what gets written to the output?             ← task-specific
W_k controls: how does this token present itself?          ← relatively stable
'''

# ── Test 1: Shape ─────────────────────────────────────────────
print("=" * 50)
print("Test 1: Output shape")
print("=" * 50)
batch, seq, d_model, rank = 2, 10, 256, 8
x    = torch.randn(batch, seq, d_model)
lora = LoRALinear(d_model, rank)
out  = lora(x)
print(f"Input  shape : {list(x.shape)}")
print(f"Output shape : {list(out.shape)}")
print(f"PASS: {list(out.shape) == [batch, seq, d_model]}")
 
 
# ── Test 2: Zero init — LoRA adds nothing at start ────────────
print()
print("=" * 50)
print("Test 2: Zero init (B=zeros → delta=0)")
print("=" * 50)
out_W    = lora.W(x)
out_lora = lora(x)
delta    = (out_lora - out_W).abs().max().item()
print(f"Max delta at init : {delta:.8f}  (should be 0.0)")
print(f"PASS: {torch.allclose(out_W, out_lora)}")
 
 
# ── Test 3: Trainable vs frozen params ────────────────────────
print()
print("=" * 50)
print("Test 3: Parameter counts")
print("=" * 50)
total     = sum(p.numel() for p in lora.parameters())
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in lora.parameters() if not p.requires_grad)
print(f"Total params      : {total:,}")
print(f"Trainable (A+B)   : {trainable:,}   ← only these get gradients")
print(f"Frozen    (W)     : {frozen:,}   ← pretrained weights locked")
print(f"Reduction         : {frozen/trainable:.1f}x fewer trainable params")
expected_trainable = 2 * d_model * rank
print(f"PASS: {trainable == expected_trainable}")
 
 
# ── Test 4: After one update — delta is non-zero ──────────────
print()
print("=" * 50)
print("Test 4: LoRA learns after one gradient step")
print("=" * 50)
optimizer = torch.optim.Adam(
    [p for p in lora.parameters() if p.requires_grad], lr=1e-3
)
target = torch.randn(batch, seq, d_model)
loss   = (lora(x) - target).pow(2).mean()
loss.backward()
optimizer.step()
 
out_after = lora(x)
delta_after = (out_after - out_W).abs().max().item()
print(f"Max delta before training : 0.0")
print(f"Max delta after 1 step    : {delta_after:.6f}  (should be > 0)")
print(f"PASS: {delta_after > 0}")
 
 
# ── Test 5: W is truly frozen — gradients only on A, B ────────
print()
print("=" * 50)
print("Test 5: Frozen weight has no gradient")
print("=" * 50)
print(f"W.weight.grad : {lora.W.weight.grad}  (should be None)")
print(f"A.grad        : {lora.A.grad is not None}  (should be True)")
print(f"B.grad        : {lora.B.grad is not None}  (should be True)")
print(f"PASS: {lora.W.weight.grad is None and lora.A.grad is not None}")
 
 
# ── Test 6: Different ranks ───────────────────────────────────
print()
print("=" * 50)
print("Test 6: Rank comparison")
print("=" * 50)
for r in [4, 8, 16, 32, 64]:
    m = LoRALinear(d_model=512, rank=r)
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    f = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"rank={r:3d}  trainable={t:7,}  frozen={f:,}  reduction={f/t:.1f}x")
 
print()
print("=" * 50)
print("All tests complete")
print("=" * 50)


batch, seq, d_model, heads, rank = 2, 10, 512, 8, 8
Q    = torch.randn(batch, seq, d_model)
K    = torch.randn(batch, seq, d_model)
V    = torch.randn(batch, seq, d_model)
mask = torch.tril(torch.ones(seq, seq))
lora_mha = LoRAMultiHeadAttention(heads=heads, d_model=d_model, rank=rank)


# ── Test 1: Output shape ──────────────────────────────────────
print("=" * 50)
print("Test 1: Output shape")
print("=" * 50)
output, weights = lora_mha(Q, K, V, mask=mask)
print(f"Input  shape   : {list(Q.shape)}")
print(f"Output shape   : {list(output.shape)}")
print(f"Weights shape  : {list(weights.shape)}")
print(f"PASS: {list(output.shape) == [batch, seq, d_model]}")


# ── Test 2: Parameter counts ──────────────────────────────────
print()
print("=" * 50)
print("Test 2: Trainable vs frozen params")
print("=" * 50)
total     = sum(p.numel() for p in lora_mha.parameters())
trainable = sum(p.numel() for p in lora_mha.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in lora_mha.parameters() if not p.requires_grad)
print(f"Total      : {total:,}")
print(f"Trainable  : {trainable:,}   ← A+B for W_q and W_v only")
print(f"Frozen     : {frozen:,}  ← W_q.W, W_k, W_v.W, W_o")
print(f"Reduction  : {frozen/trainable:.1f}x")
expected  = 2 * 2 * d_model * rank   # 2 LoRA layers × (A + B)
print(f"PASS: {trainable == expected}")


# ── Test 3: W_k and W_o truly frozen ─────────────────────────
print()
print("=" * 50)
print("Test 3: W_k and W_o are frozen")
print("=" * 50)
print(f"W_k requires_grad : {lora_mha.W_k.weight.requires_grad}  (should be False)")
print(f"W_o requires_grad : {lora_mha.W_o.weight.requires_grad}  (should be False)")
print(f"PASS: {not lora_mha.W_k.weight.requires_grad and not lora_mha.W_o.weight.requires_grad}")


# ── Test 4: W_q and W_v have trainable A and B ────────────────
print()
print("=" * 50)
print("Test 4: W_q and W_v LoRA params are trainable")
print("=" * 50)
print(f"W_q.A requires_grad : {lora_mha.W_q.A.requires_grad}  (should be True)")
print(f"W_q.B requires_grad : {lora_mha.W_q.B.requires_grad}  (should be True)")
print(f"W_v.A requires_grad : {lora_mha.W_v.A.requires_grad}  (should be True)")
print(f"W_v.B requires_grad : {lora_mha.W_v.B.requires_grad}  (should be True)")
print(f"PASS: {lora_mha.W_q.A.requires_grad and lora_mha.W_q.B.requires_grad}")


# ── Test 5: Zero init — B=zeros means no perturbation at start
print()
print("=" * 50)
print("Test 5: Zero init check")
print("=" * 50)
# manually compute W(x) without LoRA delta
out_no_lora_q = lora_mha.W_q.W(Q)
out_with_lora = lora_mha.W_q(Q)
delta = (out_with_lora - out_no_lora_q).abs().max().item()
print(f"W_q LoRA delta at init : {delta:.8f}  (should be 0.0)")
print(f"PASS: {torch.allclose(out_no_lora_q, out_with_lora)}")


# ── Test 6: Gradients flow to A and B after backward ─────────
print()
print("=" * 50)
print("Test 6: Gradients flow to A and B")
print("=" * 50)
optimizer = torch.optim.Adam(
    [p for p in lora_mha.parameters() if p.requires_grad], lr=1e-3
)
target = torch.randn(batch, seq, d_model)
loss   = (lora_mha(Q, K, V, mask=mask)[0] - target).pow(2).mean()
loss.backward()
print(f"W_q.A has grad : {lora_mha.W_q.A.grad is not None}  (should be True)")
print(f"W_q.B has grad : {lora_mha.W_q.B.grad is not None}  (should be True)")
print(f"W_k.weight.grad: {lora_mha.W_k.weight.grad}  (should be None)")
print(f"PASS: {lora_mha.W_q.A.grad is not None and lora_mha.W_k.weight.grad is None}")


# ── Test 7: Rank comparison ───────────────────────────────────
print()
print("=" * 50)
print("Test 7: Rank vs param reduction")
print("=" * 50)
print(f"{'rank':>6}  {'trainable':>12}  {'frozen':>12}  {'reduction':>10}")
for r in [4, 8, 16, 32, 64]:
    m  = LoRAMultiHeadAttention(heads=8, d_model=512, rank=r)
    t  = sum(p.numel() for p in m.parameters() if p.requires_grad)
    f  = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"{r:>6}  {t:>12,}  {f:>12,}  {f/t:>9.1f}x")

print()
print("=" * 50)
print("All tests complete")
print("=" * 50)