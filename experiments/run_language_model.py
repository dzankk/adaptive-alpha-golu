"""
Benchmark: Language Modeling & Autoregressive Generation (Mini-GPT)
Evaluates autoregressive sequence modeling perplexity across transformer block activations.
Uses Causal Multi-Head Attention with zero-weight-decay parameter splitting 
to ensure adaptive activation parameters do not decay prematurely.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Custom Activations
# ==========================================
class GoLUStatic(nn.Module):
    def forward(self, x):
        scaled = torch.clamp(-x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))

class AdaptiveAlphaGoLU(nn.Module):
    def __init__(self, init_alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        scaled = torch.clamp(-self.alpha * x, min=-88.0, max=88.0)
        return x * torch.exp(-torch.exp(scaled))

class SwishAdaptive(nn.Module):
    def __init__(self, init_beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def get_activation(act_type: str) -> nn.Module:
    act_type = act_type.lower()
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'golu_static':
        return GoLUStatic()
    elif act_type == 'alpha_golu':
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    elif act_type == 'swish_adaptive':
        return SwishAdaptive(init_beta=1.0)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")

def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'alpha' in name or 'beta' in name:
            act_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{'params': base_params, 'weight_decay': weight_decay}]
    if act_params:
        param_groups.append({'params': act_params, 'lr': lr, 'weight_decay': 0.0})

    return optim.AdamW(param_groups, lr=lr)

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
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=2, act_type='relu'):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
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
# 3. Benchmark Run
# ==========================================
class SyntheticTextDataset(Dataset):
    def __len__(self):
        return 200

    def __getitem__(self, idx):
        return torch.randint(0, 1000, (65,))

def run_lm_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Language Model Benchmark on {device}")

    dataset = SyntheticTextDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    activations = ['relu', 'gelu', 'golu_static', 'alpha_golu', 'swish_adaptive']

    for act_type in activations:
        model = MiniGPT(act_type=act_type).to(device)
        optimizer = get_optimizer(model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(2):
            for batch in loader:
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
            for batch in loader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, 1000), targets.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = math.exp(avg_loss)
        print(f"Activation: {act_type.ljust(15)} | Test Perplexity: {perplexity:.2f}")

def train_and_eval(activation: str = 'alpha_golu', seed: int = 42, epochs: int = 10) -> float:
    """Returns Perplexity (PPL)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppl = train_single_seed_lm(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(ppl)
    
if __name__ == '__main__':
    run_lm_benchmark()
