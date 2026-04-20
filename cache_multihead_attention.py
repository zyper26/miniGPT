import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_components import scaled_dot_product_attention

class CachedMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        assert d_model % heads == 0
        self.h   = heads
        self.d_k = d_model // heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        q_projection = self.W_q(x)
        k_projection = self.W_k(x)
        v_projection = self.W_v(x)

        batch, seq, d_model = q_projection.shape
        
        if not use_cache:
            q, k, v = q_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2), \
                k_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2), \
                v_projection.view(batch, seq, self.h, self.d_k).transpose(1, 2)
            self.cache_k = k
            self.cache_v = v
        else:
            new_k = k_projection.view(batch, 1, self.h, self.d_k).transpose(1, 2)
            new_v = v_projection.view(batch, 1, self.h, self.d_k).transpose(1, 2)

            k, v = torch.concat([self.cache_k, new_k], dim=2), torch.concat([self.cache_v, new_v], dim=2)
            self.cache_k = k
            self.cache_v = v

            q = q_projection.view(batch, 1, self.h, self.d_k).transpose(1, 2)
        
        output, weights = scaled_dot_product_attention(q, k, v)

        concatenated_output = output.transpose(1,2).contiguous().view(batch, seq, self.h*self.d_k)
        return self.W_o(concatenated_output), weights

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None


if __name__ == "__main__":
    batch, seq, d_model, heads = 1, 10, 256, 4
    mha = CachedMultiHeadAttention(heads=heads, d_model=d_model)
 
    prompt = torch.randn(batch, seq, d_model)
    out, _ = mha(prompt, use_cache=False)
    print(f"Prefill output : {list(out.shape)}")
    print(f"Cache K shape  : {list(mha.cache_k.shape)}")
 
    for step in range(3):
        token = torch.randn(batch, 1, d_model)
        out, _ = mha(token, use_cache=True)
        print(f"Decode step {step+1}  : output={list(out.shape)}  cache_k={list(mha.cache_k.shape)}")
 
    print(f"\nCache grew correctly: {seq} → {seq+3} tokens")
 
    L, h, t, d_k = 32, heads, seq+3, d_model//heads
    bytes_per_el  = 2   # float16
    cache_gb      = 2 * L * batch * h * t * d_k * bytes_per_el / 1e9
    print(f"\nKV cache memory ({L} layers, t={t}): {cache_gb*1000:.2f} MB")