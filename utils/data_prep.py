"""
Dataset preparation helpers for long-running benchmark experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection, VOCSegmentation, CelebA


def _download_cifar10(root: Path) -> None:
    torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=transforms.ToTensor())
    torchvision.datasets.CIFAR10(root=str(root), train=False, download=True, transform=transforms.ToTensor())


def _download_fashion_mnist(root: Path) -> None:
    torchvision.datasets.FashionMNIST(root=str(root), train=True, download=True, transform=transforms.ToTensor())
    torchvision.datasets.FashionMNIST(root=str(root), train=False, download=True, transform=transforms.ToTensor())


def _download_voc(root: Path) -> None:
    VOCDetection(root=str(root), year="2012", image_set="trainval", download=True)
    VOCDetection(root=str(root), year="2012", image_set="val", download=True)
    VOCSegmentation(root=str(root), year="2012", image_set="train", download=True)
    VOCSegmentation(root=str(root), year="2012", image_set="val", download=True)


def _download_celeba(root: Path) -> None:
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    CelebA(root=str(root), split="train", target_type="attr", download=True, transform=transform)
    CelebA(root=str(root), split="valid", target_type="attr", download=True, transform=transform)


def _download_wikitext2(root: Path) -> None:
    from experiments.run_language_model import download_wikitext2

    download_wikitext2(root=str(root))


def prepare_all_datasets(root: str = "./data") -> Dict[str, List[str]]:
    """Download all datasets used by the benchmark suite once and reuse them afterward."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    actions = [
        ("cifar10", _download_cifar10),
        ("fashion_mnist", _download_fashion_mnist),
        ("pascal_voc", _download_voc),
        ("celeba", _download_celeba),
        ("wikitext2", _download_wikitext2),
    ]

    completed: List[str] = []
    for name, action in actions:
        action(root_path)
        completed.append(name)

    return {"root": str(root_path), "datasets": completed}