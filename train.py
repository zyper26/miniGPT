import torch
import torch.nn.functional as F
import torch.nn as nn
from transformer_components import *
import math

def train(model, tokens, epoch=10, lr=1e-3):
    input_    = tokens[:, :-1]
    target    = tokens[:, 1:].reshape(-1)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    for _ in range(epoch):
        optimizer.zero_grad()
        output = model(input_)                     
        output = output.reshape(-1, output.size(-1))
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"epoch {_+1:3d}  loss: {loss.item():.4f}")
 
    return loss
 
 
if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size, d_model, heads, num_layers = 256, 128, 4, 2
    batch, seq = 4, 32
 
    tokens = torch.randint(0, vocab_size, (batch, seq))
    model  = MiniGPT(vocab_size=vocab_size, d_model=d_model,
                     heads=heads, num_layers=num_layers)
 
    print(f"Params          : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Random baseline : {math.log(vocab_size):.4f}")
    print()
    train(model, tokens, epoch=10)
 