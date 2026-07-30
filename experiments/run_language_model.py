"""
Benchmark: Language Modeling & Autoregressive Generation (Mini-GPT)
===================================================================
Evaluates autoregressive sequence modeling perplexity across transformer 
block activations (ReLU, GELU, Swish, PReLU, PGELU, Static GoLU, 
Adaptive Alpha-GoLU, and Adaptive Swish).

Uses Causal Multi-Head Attention with zero-weight-decay parameter splitting 
to ensure adaptive activation parameters do not decay prematurely.
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
except ImportError:
    class StaticGoLU(nn.Module):
        """Static Gompertz Linear Unit: x * exp(-exp(-x))"""
        def forward(self, x):
            scaled = torch.clamp(-x, min=-88.0, max=88.0)
            return x * torch.exp(-torch.exp(scaled))

    class AdaptiveAlphaGoLU(nn.Module):
        """Adaptive Gompertz Linear Unit: x * exp(-exp(-alpha * x)) with softplus safety constraint"""
        def __init__(self, init_alpha=1.0):
            super().__init__()
            init_val = float(init_alpha)
            init_raw = math.log(math.exp(init_val) - 1.0) if init_val < 20 else init_val
            self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

        @property
        def alpha(self):
            return nn.functional.softplus(self.raw_alpha)

        def forward(self, x):
            scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
            return x * torch.exp(-torch.exp(scaled))


def reset_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. Custom Activations
# ==========================================
class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x) with softplus constraint"""
    def __init__(self, init_alpha=1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.exp(init_val) - 1.0) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        return nn.functional.softplus(self.raw_alpha)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    """Adaptive Swish (SiLU): x * sigmoid(beta * x)"""
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type in ('swish', 'silu'):
        return nn.SiLU()
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'pgelu':
        return PGELU(init_alpha=1.0)
    elif act_type in ('swish_adaptive', 'adaptive_swish'):
        return AdaptiveSwish(init_beta=1.0)
    elif act_type == 'golu_static':
        return StaticGoLU()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    
    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, PGELU, AdaptiveSwish, nn.PReLU)):
            for p in module.parameters():
                if p.requires_grad:
                    act_params.append(p)

    act_param_ids = set(map(id, act_params))
    for p in model.parameters():
        if p.requires_grad and id(p) not in act_param_ids:
            base_params.append(p)

    param_groups = [{'params': base_params, 'lr': lr, 'weight_decay': weight_decay}]
    if act_params:
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups)


# ==========================================
# 2. Mini GPT Architecture
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        context = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, act_type):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            get_activation(act_type),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=2, max_seq_len=512, act_type='relu'):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, act_type) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ==========================================
# 3. Structured Synthetic Dataset
# ==========================================
class SyntheticTextDataset(Dataset):
    """
    Generates synthetic token sequences with local correlations 
    to evaluate perplexity meaningfully.
    """
    def __init__(self, vocab_size=1000, seq_len=65, num_samples=500):
        self.samples = []
        for i in range(num_samples):
            pattern_len = 8
            pattern = torch.randint(0, vocab_size // 10, (pattern_len,))
            seq = pattern.repeat((seq_len // pattern_len) + 1)[:seq_len]
            noise_mask = torch.rand(seq_len) < 0.10
            seq[noise_mask] = torch.randint(0, vocab_size, (noise_mask.sum().item(),))
            self.samples.append(seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================
def train_single_seed_lm(act_type='alpha_golu', seed=42, epochs=10, device='cuda'):
    reset_all_seeds(seed)
    dataset = SyntheticTextDataset(num_samples=500)
    
    # Proper 80/20 train/test split to evaluate generalization perplexity
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = Subset(dataset, range(0, train_size)), Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = MiniGPT(act_type=act_type).to(device)
    optimizer = get_optimizer(model, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, 1000), targets.reshape(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, 1000), targets.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    
    try:
        perplexity = math.exp(min(avg_loss, 20.0))
    except OverflowError:
        perplexity = float('inf')
        
    return perplexity


def run_lm_benchmark(seeds=[42, 123, 999, 2024, 2025], epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Language Model Benchmark on {device}...")
    activations = ['relu', 'gelu', 'swish', 'adaptive_swish', 'prelu', 'pgelu', 'golu_static', 'alpha_golu']

    print("\n================ Mini-GPT Language Model Test Perplexity (PPL ↓) ================")
    for act_type in activations:
        ppls = []
        for s in seeds:
            ppl = train_single_seed_lm(act_type=act_type, seed=s, epochs=epochs, device=device)
            ppls.append(ppl)
            print(f"[{act_type.upper():<14} | Seed {s}] Test Perplexity: {ppl:.2f}")

        print(f"  --> {act_type.upper():<14} Mean PPL: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}\n")


def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Test Perplexity (PPL)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl = train_single_seed_lm(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(ppl)


if __name__ == '__main__':
    run_lm_benchmark()
