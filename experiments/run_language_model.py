"""
Benchmark: WikiText-2 Word-Level Language Modeling
===================================================
Trains a compact causal transformer on a real WikiText-2 corpus and reports
validation perplexity.

This runner is grounded in a standard language modeling benchmark rather than a
synthetic proxy dataset.
"""

import math
import sys
import random
import re
import urllib.request
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json
from utils.overhead_tracker import OverheadTracker
from utils.train_tuning import bf16_autocast, build_adamw_with_activation_groups, clip_activation_gradients, configure_benchmark_runtime, default_loader_kwargs, resolve_task_alpha_hparams


WIKITEXT2_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def reset_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_benchmark_runtime()


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PGELU(nn.Module):
    """Parametric GELU: x * CDF(alpha * x) with softplus-constrained alpha."""

    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self):
        return nn.functional.softplus(self.raw_alpha)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class AdaptiveSwish(nn.Module):
    """Parametric Swish (SiLU): x * sigmoid(beta * x)."""

    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def get_activation(act_type: str) -> nn.Module:
    act_type = str(act_type).lower().strip()
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "gelu":
        return nn.GELU()
    if act_type in ("swish", "silu"):
        return nn.SiLU()
    if act_type == "prelu":
        return nn.PReLU()
    if act_type == "pgelu":
        return PGELU(init_alpha=1.0)
    if act_type in ("swish_adaptive", "adaptive_swish"):
        return AdaptiveSwish(init_beta=1.0)
    if act_type == "golu_static":
        return StaticGoLU()
    if act_type == "alpha_golu":
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    alpha_lr: float | None = None,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 1,
):
    return build_adamw_with_activation_groups(
        model,
        base_lr=lr,
        base_weight_decay=weight_decay,
        activation_lr=alpha_lr,
        activation_weight_decay=0.0,
        warmup_epochs=warmup_epochs,
    )


def download_wikitext2(root: str = "./data") -> Dict[str, Path]:
    corpus_dir = Path(root) / "wikitext-2"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, url in WIKITEXT2_URLS.items():
        path = corpus_dir / f"{split}.txt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        paths[split] = path
    return paths


def basic_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_corpus(path: Path) -> List[str]:
    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tokens.extend(basic_tokenize(stripped))
            tokens.append("<eos>")
    return tokens


def build_vocab(token_lists: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for token, frequency in counter.items():
        if frequency >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> torch.Tensor:
    unk_id = vocab["<unk>"]
    return torch.tensor([vocab.get(token, unk_id) for token in tokens], dtype=torch.long)


class BlockDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int = 64, stride: int | None = None):
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.blocks = []

        for start in range(0, max(0, len(token_ids) - block_size - 1), self.stride):
            self.blocks.append(token_ids[start : start + block_size + 1])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def collate_blocks(batch):
    return torch.stack(batch, dim=0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, time_steps, channels = x.size()
        qkv = self.qkv(x).reshape(batch_size, time_steps, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device)).view(1, 1, time_steps, time_steps)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        context = (attention @ v).transpose(1, 2).reshape(batch_size, time_steps, channels)
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
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=2, max_seq_len=256, act_type="relu"):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, act_type) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        _, time_steps = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :time_steps, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def build_language_model_dataloaders(
    dataset_name: str = "wikitext2",
    root: str = "./data",
    block_size: int = 64,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    normalized_name = str(dataset_name).lower().strip()
    if normalized_name not in {"wikitext2", "wiki_text2", "wiki-text-2"}:
        raise ValueError(f"Unsupported language dataset: {dataset_name}")

    paths = download_wikitext2(root=root)
    train_tokens = read_corpus(paths["train"])
    valid_tokens = read_corpus(paths["valid"])
    vocab = build_vocab([train_tokens])

    train_ids = encode_tokens(train_tokens, vocab)
    valid_ids = encode_tokens(valid_tokens, vocab)

    train_dataset = BlockDataset(train_ids, block_size=block_size)
    valid_dataset = BlockDataset(valid_ids, block_size=block_size)

    loader_g = torch.Generator().manual_seed(seed)
    train_loader_kwargs = default_loader_kwargs()
    valid_loader_kwargs = default_loader_kwargs(num_workers=1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_g,
        worker_init_fn=seed_worker,
        collate_fn=collate_blocks,
        **train_loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=loader_g,
        worker_init_fn=seed_worker,
        collate_fn=collate_blocks,
        **valid_loader_kwargs,
    )
    return train_loader, valid_loader, vocab


def train_single_seed_lm(
    act_type: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 5,
    device: str = "cuda",
    dataset_name: str = "wikitext2",
    block_size: int = 64,
    data_root: str = "./data",
    alpha_lr: float | None = None,
    config_path: str | None = "configs/paper_benchmark.json",
    save_artifacts: bool = False,
    amp: bool = False,
) -> float:
    reset_all_seeds(seed)

    train_loader, valid_loader, vocab = build_language_model_dataloaders(
        dataset_name=dataset_name,
        root=data_root,
        block_size=block_size,
        batch_size=32,
        seed=seed,
    )

    model = MiniGPT(vocab_size=len(vocab), act_type=act_type, max_seq_len=block_size).to(device)
    overhead_tracker = OverheadTracker(task_name="language_model", activation_name=act_type, model=model, device=device)
    alpha_lr, alpha_warmup_epochs, alpha_grad_clip_norm = resolve_task_alpha_hparams(
        "language_model",
        alpha_lr,
        config_path=config_path,
    )
    optimizer, set_alpha_lr, act_params = get_optimizer(model, lr=1e-3, alpha_lr=alpha_lr, warmup_epochs=alpha_warmup_epochs)
    criterion = nn.CrossEntropyLoss()
    alpha_logger = AlphaTrajectoryLogger(model)
    train_start = time.perf_counter()
    epoch_seconds = []
    amp_enabled = bool(amp) and torch.cuda.is_available() and "cuda" in str(device)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        current_alpha_lr = set_alpha_lr(epoch)
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            overhead_tracker.start_forward()
            with bf16_autocast(amp_enabled):
                logits = model(inputs)
            overhead_tracker.end_forward(batch_size=inputs.size(0))
            loss = criterion(logits.float().reshape(-1, len(vocab)), targets.reshape(-1))

            optimizer.zero_grad()
            overhead_tracker.start_backward()
            loss.backward()
            overhead_tracker.end_backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            clip_activation_gradients(model, max_norm=alpha_grad_clip_norm)
            optimizer.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        with bf16_autocast(amp_enabled):
            for batch in valid_loader:
                batch = batch.to(device, non_blocking=True)
                inputs, targets = batch[:, :-1], batch[:, 1:]
                logits = model(inputs)
                loss = criterion(logits.float().reshape(-1, len(vocab)), targets.reshape(-1))
                total_loss += float(loss.item()) * targets.numel()
                total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    overhead = overhead_tracker.save()

    if save_artifacts:
        run_dir = create_run_directory(
            str(PROJECT_ROOT / "outputs" / "runs" / "language_model"),
            "language_model",
            act_type,
            [seed],
        )
        write_json(
            run_dir / "results.json",
            {
                "task": "language_model",
                "dataset_name": dataset_name,
                "data_root": data_root,
                "activation": act_type,
                "alpha_lr": alpha_lr,
                "seed": seed,
                "epochs": epochs,
                "block_size": block_size,
                "alpha_lr_final": current_alpha_lr if act_params else None,
                "perplexity": float(perplexity),
                "avg_loss": float(avg_loss),
                "alpha_history": alpha_logger.alpha_history,
                **overhead,
            },
        )
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=f"python {Path(__file__).name} --activation {act_type} --seeds {seed} --epochs {epochs}",
                task="language_model",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "dataset_name": dataset_name,
                    "data_root": data_root,
                    "epochs": epochs,
                    "alpha_lr": alpha_lr if alpha_lr is not None else 1e-3,
                    "seed": seed,
                    "block_size": block_size,
                },
            ),
        )
        if alpha_logger.alpha_history:
            alpha_logger.plot_trajectories(str(run_dir / "alpha_trajectories.png"))

    return float(perplexity)


def run_lm_benchmark(seeds=None, epochs=5, data_root: str = "./data", alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", amp: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 999, 2024, 2025]
    print(f"Running WikiText-2 Language Model Benchmark on {device}...")
    activations = ["relu", "gelu", "swish", "adaptive_swish", "prelu", "pgelu", "golu_static", "alpha_golu"]

    print("\n================ WikiText-2 Language Model Test Perplexity (PPL ↓) ================")
    for act_type in activations:
        ppls = []
        for s in seeds:
            ppl = train_single_seed_lm(act_type=act_type, seed=s, epochs=epochs, device=device, data_root=data_root, alpha_lr=alpha_lr, config_path=config_path, save_artifacts=True, amp=amp)
            ppls.append(ppl)
            print(f"[{act_type.upper():<14} | Seed {s}] Validation Perplexity: {ppl:.2f}")

        print(f"  --> {act_type.upper():<14} Mean PPL: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}\n")


def train_and_eval(activation: str = "alpha_golu", seed: int = 42, epochs: int = 5, data_root: str = "./data", alpha_lr: float | None = None, config_path: str | None = "configs/paper_benchmark.json", save_artifacts: bool = False, amp: bool = False) -> float:
    """Returns validation perplexity on WikiText-2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perplexity = train_single_seed_lm(act_type=activation, seed=seed, epochs=epochs, device=device, data_root=data_root, alpha_lr=alpha_lr, config_path=config_path, save_artifacts=save_artifacts, amp=amp)
    return float(perplexity)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WikiText-2 language modeling benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Single activation to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999, 2024, 2025], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--alpha-lr", type=float, default=None, help="Learning rate for language-model activation parameters; defaults to configs/paper_benchmark.json")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.json", help="Benchmark config file used for task alpha hyperparameters")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep")
    parser.add_argument("--amp", action="store_true", help="Enable BF16 automatic mixed precision on CUDA")
    args = parser.parse_args()

    if args.benchmark:
        run_lm_benchmark(seeds=args.seeds, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running WikiText-2 Language Model Benchmark on {device}...")
        for seed in args.seeds:
            ppl = train_and_eval(activation=args.activation, seed=seed, epochs=args.epochs, data_root=args.data_root, alpha_lr=args.alpha_lr, config_path=args.config, amp=args.amp)
            print(f"[{args.activation.upper():<14} | Seed {seed}] Validation Perplexity: {ppl:.2f}")