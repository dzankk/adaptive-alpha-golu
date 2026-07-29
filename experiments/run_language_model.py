"""
Mini-GPT Language Modeling Benchmark
====================================
Evaluates GELU, Static GoLU, and Adaptive Alpha-GoLU on a 4-layer
Transformer Decoder (Language Modeling task on Shakespeare dataset).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import urllib.request

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU
except ImportError:
    from models.alpha_golu import AlphaGoLUModule as AdaptiveAlphaGoLU


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, act_type='alpha_golu'):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # MLP Block with customizable activation
        if act_type == 'gelu':
            act = nn.GELU()
        elif act_type == 'golu_static':
            class StaticGoLU(nn.Module):
                def forward(self, x):
                    return x * torch.exp(-torch.exp(-x))
            act = StaticGoLU()
        elif act_type == 'alpha_golu':
            act = AdaptiveAlphaGoLU()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            act,
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layer=3, act_type='alpha_golu'):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, act_type) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def run_lm_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Language Modeling (Mini-GPT) Benchmark on {device}...")

    # Download tiny Shakespeare text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    try:
        with urllib.request.urlopen(url) as f:
            text = f.read().decode('utf-8')[:50000] # Use first 50k chars for speed
    except Exception:
        text = "To be or not to be, that is the question. " * 1000

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    # Train / Test split
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split='train', batch_size=32, block_size=64):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    results = {}
    seeds = [42, 123]

    print("\n================ Mini-GPT Language Modeling (Perplexity ↓) ================")
    for act_type in ['gelu', 'golu_static', 'alpha_golu']:
        perplexities = []
        for s in seeds:
            torch.manual_seed(s)
            model = MiniGPT(vocab_size, act_type=act_type).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            model.train()
            for step in range(300): # Fast 300 steps
                xb, yb = get_batch('train')
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loss / perplexity
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):
                    xb, yb = get_batch('val')
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                    val_losses.append(loss.item())
            
            mean_val_loss = np.mean(val_losses)
            ppl = math.exp(mean_val_loss)
            perplexities.append(ppl)
            print(f"[{act_type.upper():<12} | Seed {s}] Val Loss: {mean_val_loss:.4f} | Perplexity: {ppl:.2f}")

        results[act_type] = (np.mean(perplexities), np.std(perplexities))

    print("\n================ LANGUAGE MODELING SUMMARY ================")
    for act_type, (mean_ppl, std_ppl) in results.items():
        print(f"  {act_type.upper():<12}: Perplexity = {mean_ppl:.2f} ± {std_ppl:.2f}")

if __name__ == '__main__':
    run_lm_benchmark()
