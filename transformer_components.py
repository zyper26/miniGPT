import torch
import torch.nn.functional as F
import torch.nn as nn

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    qk = torch.matmul(Q, K.transpose(-2, -1))
    
    normalized_qk = qk/d_k**0.5
    
    if mask is not None:
        normalized_qk = normalized_qk.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(normalized_qk, dim=-1)
    output = torch.matmul(weights, V)

    return output, weights


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model):
        super().__init__()
        assert d_model%heads == 0
        self.h = heads
        self.d_k = d_model//heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        q_projection = self.W_q(Q)
        k_projection = self.W_k(K)
        v_projection = self.W_v(V)

        batch = Q.size(0)
        seq   = Q.size(1)
        q, k, v = q_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2), \
                k_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2), \
                v_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2)

        outputs, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        concatenated = outputs.transpose(1, 2).contiguous().view(batch, seq, Q.size(-1))

        return self.W_o(concatenated), weights


batch, seq_len, d_model, h = 2, 10, 256, 8

Q = torch.randn(batch, seq_len, d_model)
K = torch.randn(batch, seq_len, d_model)
V = torch.randn(batch, seq_len, d_model)

mask = torch.tril(torch.ones(seq_len, seq_len))

mha = MultiHeadAttention(heads=h, d_model=d_model)
output, weights = mha(Q, K, V, mask=mask)

print(f"Input Q shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Weights shape: {weights.shape}")

'''batch=2 — we passed 2 sentences at once.
heads=8 — each of the 8 heads produces its own attention weight matrix independently.
seq × seq = 10 × 10 — this is the interesting part. For every token in the sequence, we compute a score against every other token. So for a sequence of 10 tokens you get a 10×10 grid:
         token1  token2  token3 ... token10
token1  [ 0.8    0.1     0.05  ...  0.0  ]   ← how much token1 attends to each other token
token2  [ 0.2    0.6     0.1   ...  0.1  ]
token3  [ 0.0    0.3     0.5   ...  0.2  ]
...
token10 [ 0.1    0.0     0.2   ...  0.7  ]
Each row sums to 1.0 (because of softmax). Each row answers the question: "for this token, how much should I look at each other token?"
This is why attention is O(n²) in memory and compute — the weight matrix grows quadratically with sequence length. A 1000-token sequence gives a 1000×1000 matrix per head. This is the core bottleneck that papers like FlashAttention and Longformer were designed to solve.
With the causal mask applied (lower triangular), the upper triangle becomes 0 — token 3 can't attend to token 5, only to tokens 1, 2, 3. That's what makes it a decoder style attention.'''



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model,))
        self.beta = nn.Parameter(torch.zeros(d_model,))
        self.eps = eps

    def forward(self, x):
        input_mean = x.mean(dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        normalized = (x - input_mean)/(variance + self.eps)**0.5
        return self.gamma*normalized + self.beta

x = torch.randn(batch, seq_len, d_model)

ln = LayerNorm(d_model)
layer_norm_output = ln(x)

print(f"Layer Norm Shape: {layer_norm_output.shape}")

'''Mean ≈ 0, Std ≈ 1 — normalization is working exactly as intended.'''

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        self.d_ff = 4*d_model
        self.linear1 = nn.Linear(d_model, self.d_ff, bias = False)
        self.linear2 = nn.Linear(self.d_ff, d_model, bias = False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

x = torch.randn(batch, seq_len, d_model)
ff = FeedForward(d_model)
ff_output = ff(x)

print(f"FF Output Shape: {ff_output.shape}")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff=None):
        super().__init__()
        self.mha = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ln1 = LayerNorm(d_model=d_model)
        self.ln2 = LayerNorm(d_model=d_model)
        self.ff = FeedForward(d_model=d_model)

    def forward(self, x, mask=None):
        mha_out, _ = self.mha(x, x, x, mask=mask)
        x = self.ln1(x + mha_out)
        x = self.ln2(x + self.ff(x))
        return x

x = torch.randn(batch, seq_len, d_model)
tb = TransformerBlock(d_model=d_model, heads=h)
tb_output = tb(x)

print(f"Transformer Block Shape: {tb_output.shape}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        denominator = torch.pow(10000, (i/d_model))

        pe[:, 0::2] = torch.sin(pos/denominator)
        pe[:, 1::2] = torch.cos(pos/denominator)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1),:].unsqueeze(dim=0) + x

x = torch.randn(batch, seq_len, d_model)
poe = PositionalEncoding(d_model=d_model)
poe_output = poe(x)

print(f"Positional Encoding Shape: {poe_output.shape}")

'''
tokens (batch, seq)
    ↓  Embedding          → (batch, seq, d_model)
    ↓  PositionalEncoding → (batch, seq, d_model)
    ↓  TransformerBlock × N
    ↓  Linear projection  → (batch, seq, vocab_size)
    ↓  logits
'''


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, heads, num_layers, max_seq_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.num_layers = num_layers
        self.transformer = nn.ModuleList([TransformerBlock(d_model=d_model, heads=heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        seq = x.size(1)
        mask = torch.tril(torch.ones(seq, seq, device=x.device))
        for block in self.transformer:
            x = block(x, mask=mask)
        x = self.linear(x)
        return x

vocab_size, num_layers = 50257, 6
tokens = torch.randint(0, vocab_size, (batch, seq_len))
mGPT = MiniGPT(vocab_size=vocab_size, d_model=d_model, heads=h, num_layers=num_layers)
logits = mGPT(tokens)

total_params = sum(p.numel() for p in mGPT.parameters())
embedding_params = sum(p.numel() for p in mGPT.embedding.parameters())
transformer_params = sum(p.numel() for p in mGPT.transformer.parameters())
linear_params = sum(p.numel() for p in mGPT.linear.parameters())

print(f"Output: {logits.shape}")
print(f"Total Model Parameters: {total_params:,}")
print(f"Embedding Parameters: {embedding_params:,}")
print(f"Transformer Parameters: {transformer_params:,}")
print(f"Linear Parameters: {linear_params:,}")
