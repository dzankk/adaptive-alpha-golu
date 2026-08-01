"""
Benchmark: Pascal VOC 2012 Object Detection
===========================================
Trains a standard Faster R-CNN baseline on real Pascal VOC annotations and
reports VOC-style mAP@0.5 on a held-out validation split.

The detection head uses a selectable activation function, while any learnable
activation scalars remain isolated from weight decay.
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VOCDetection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU
from diagnostics.trajectory_logger import AlphaTrajectoryLogger
from utils.run_artifacts import build_run_manifest, create_run_directory, write_json


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {name: index + 1 for index, name in enumerate(VOC_CLASSES)}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PGELU(nn.Module):
    """Parametric GELU: x * Phi(alpha * x) with positive alpha."""

    def __init__(self, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf((self.alpha * x) / 1.41421356237))


class SwishAdaptive(nn.Module):
    """Parametric Swish (SiLU): x * sigmoid(beta * x)."""

    def __init__(self, init_beta: float = 1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


AdaptiveSwish = SwishAdaptive


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
    if act_type == "golu_static":
        return StaticGoLU()
    if act_type == "alpha_golu":
        return AdaptiveAlphaGoLU(init_alpha=1.0)
    if act_type in ("swish_adaptive", "adaptive_swish"):
        return SwishAdaptive(init_beta=1.0)
    raise ValueError(f"Unknown activation type: {act_type}")


def get_optimizer(model: nn.Module, lr: float = 2e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    act_params = []
    base_params = []

    for module in model.modules():
        if isinstance(module, (AdaptiveAlphaGoLU, PGELU, SwishAdaptive, AdaptiveSwish, nn.PReLU)):
            for parameter in module.parameters(recurse=False):
                if parameter.requires_grad:
                    act_params.append(parameter)

    act_param_ids = set(map(id, act_params))
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in act_param_ids:
            base_params.append(parameter)

    parameter_groups = [{"params": base_params, "weight_decay": weight_decay}]
    if act_params:
        parameter_groups.append({"params": act_params, "lr": lr, "weight_decay": 0.0})

    return optim.AdamW(parameter_groups, lr=lr)


class PascalVOCDataset(Dataset):
    """Pascal VOC wrapper returning resized images and parsed annotations."""

    def __init__(
        self,
        root: str = "./data",
        year: str = "2012",
        image_set: str = "trainval",
        image_size: int = 320,
        download: bool = True,
        max_samples: int | None = None,
    ):
        self.dataset = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.image_size = image_size
        self.max_samples = max_samples

    def __len__(self) -> int:
        length = len(self.dataset)
        return min(length, self.max_samples) if self.max_samples is not None else length

    def _ensure_list(self, objects):
        if objects is None:
            return []
        if isinstance(objects, list):
            return objects
        return [objects]

    def __getitem__(self, idx: int):
        image, annotation = self.dataset[idx]
        annotation = annotation["annotation"]
        objects = self._ensure_list(annotation.get("object", []))

        width = float(annotation["size"]["width"])
        height = float(annotation["size"]["height"])

        boxes = []
        labels = []
        for obj in objects:
            if int(obj.get("difficult", 0)) == 1:
                continue
            class_name = str(obj["name"]).lower().strip()
            if class_name not in VOC_CLASS_TO_IDX:
                continue

            bbox = obj["bndbox"]
            x1 = float(bbox["xmin"]) - 1.0
            y1 = float(bbox["ymin"]) - 1.0
            x2 = float(bbox["xmax"]) - 1.0
            y2 = float(bbox["ymax"]) - 1.0
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(VOC_CLASS_TO_IDX[class_name])

        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        scale_x = self.image_size / width
        scale_y = self.image_size / height
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.long),
        }
        return image, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class ActivationBoxHead(nn.Module):
    """Two-layer FC box head with a selectable activation."""

    def __init__(self, in_channels: int, representation_size: int, act_type: str):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.act6 = get_activation(act_type)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.act7 = get_activation(act_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.act6(self.fc6(x))
        x = self.act7(self.fc7(x))
        return x


def build_detection_model(act_type: str, num_classes: int = 21) -> FasterRCNN:
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=3)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=None,
    )

    representation_size = 1024
    in_channels = model.roi_heads.box_head.fc6.in_features
    model.roi_heads.box_head = ActivationBoxHead(in_channels, representation_size, act_type)
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes)
    return model


def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(mpre.size - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def evaluate_map50(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.05,
) -> float:
    model.eval()

    detections_by_class: Dict[int, List[Tuple[float, int, torch.Tensor]]] = defaultdict(list)
    ground_truth_by_class: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in dataloader:
            batch_images = [image.to(device) for image in images]
            predictions = model(batch_images)

            for batch_index, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    ground_truth_by_class[int(gt_label.item())][image_id].append(gt_box.clone())

                prediction = predictions[batch_index]
                keep = prediction["scores"] >= score_thresh
                prediction = {
                    "boxes": prediction["boxes"][keep].detach().cpu(),
                    "scores": prediction["scores"][keep].detach().cpu(),
                    "labels": prediction["labels"][keep].detach().cpu(),
                }

                for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
                    detections_by_class[int(label.item())].append((float(score.item()), image_id, box.detach().cpu()))

    ap_values = []
    for class_index in range(1, len(VOC_CLASSES) + 1):
        gt_for_class = ground_truth_by_class.get(class_index, {})
        total_gt = sum(len(boxes) for boxes in gt_for_class.values())
        if total_gt == 0:
            continue

        detections = sorted(detections_by_class.get(class_index, []), key=lambda item: item[0], reverse=True)
        true_positives = []
        false_positives = []
        matched = {
            image_id: torch.zeros(len(boxes), dtype=torch.bool)
            for image_id, boxes in gt_for_class.items()
        }

        for score, image_id, pred_box in detections:
            gt_boxes = gt_for_class.get(image_id, [])
            if not gt_boxes:
                true_positives.append(0)
                false_positives.append(1)
                continue

            gt_tensor = torch.stack([box.to(pred_box.device) for box in gt_boxes])
            ious = box_iou(pred_box.unsqueeze(0), gt_tensor).squeeze(0)
            best_iou, best_index = torch.max(ious, dim=0)

            if float(best_iou.item()) >= 0.5 and not matched[image_id][best_index]:
                true_positives.append(1)
                false_positives.append(0)
                matched[image_id][best_index] = True
            else:
                true_positives.append(0)
                false_positives.append(1)

        if not true_positives:
            ap_values.append(0.0)
            continue

        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        recalls = tp_cumsum / max(total_gt, 1)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
        ap_values.append(voc_ap(recalls.astype(np.float32), precisions.astype(np.float32)))

    return float(np.mean(ap_values)) if ap_values else 0.0


def train_single_seed_detection(
    act_type: str,
    seed: int,
    epochs: int,
    device: torch.device,
    data_root: str = "./data",
    lr: float = 2e-4,
    image_size: int = 320,
    train_split_ratio: float = 0.9,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    save_artifacts: bool = False,
) -> float:
    set_seed(seed)

    full_dataset = PascalVOCDataset(
        root=data_root,
        year="2012",
        image_set="trainval",
        image_size=image_size,
        download=True,
    )

    total_length = len(full_dataset)
    if max_train_samples is not None or max_eval_samples is not None:
        train_length = max_train_samples if max_train_samples is not None else int(total_length * train_split_ratio)
        eval_length = max_eval_samples if max_eval_samples is not None else max(1, total_length - train_length)
        train_length = min(train_length, total_length)
        eval_length = min(eval_length, max(1, total_length - train_length))
    else:
        train_length = int(total_length * train_split_ratio)
        eval_length = total_length - train_length

    generator = torch.Generator().manual_seed(seed)
    train_dataset, eval_dataset = random_split(full_dataset, [train_length, eval_length], generator=generator)

    loader_g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        generator=loader_g,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker,
        generator=loader_g,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_detection_model(act_type=act_type, num_classes=len(VOC_CLASSES) + 1).to(device)
    optimizer = get_optimizer(model, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(1, epochs // 2), max(1, (3 * epochs) // 4)], gamma=0.1)
    alpha_logger = AlphaTrajectoryLogger(model)
    train_start = time.perf_counter()
    epoch_seconds = []

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device),
                    "image_id": target["image_id"].to(device),
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)
        alpha_logger.step()

    train_seconds = time.perf_counter() - train_start

    map50 = evaluate_map50(model, eval_loader, device=device)

    if save_artifacts:
        run_dir = create_run_directory(
            str(PROJECT_ROOT / "outputs" / "runs" / "detection"),
            "detection",
            act_type,
            [seed],
        )
        write_json(
            run_dir / "results.json",
            {
                "task": "detection",
                "dataset_name": "pascal_voc_2012",
                "data_root": data_root,
                "activation": act_type,
                "seed": seed,
                "epochs": epochs,
                "lr": lr,
                "image_size": image_size,
                "train_split_ratio": train_split_ratio,
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples,
                "map50": float(map50),
                "train_seconds": train_seconds,
                "epoch_seconds": epoch_seconds,
                "alpha_history": alpha_logger.alpha_history,
            },
        )
        write_json(
            run_dir / "run_manifest.json",
            build_run_manifest(
                command=(
                    f"python {Path(__file__).name} --activation {act_type} --seeds {seed} "
                    f"--epochs {epochs} --lr {lr}"
                ),
                task="detection",
                seeds=[seed],
                activations=[act_type],
                extra_config={
                    "dataset_name": "pascal_voc_2012",
                    "data_root": data_root,
                    "epochs": epochs,
                    "seed": seed,
                    "lr": lr,
                    "image_size": image_size,
                    "train_split_ratio": train_split_ratio,
                    "max_train_samples": max_train_samples,
                    "max_eval_samples": max_eval_samples,
                },
            ),
        )
        if alpha_logger.alpha_history:
            alpha_logger.plot_trajectories(str(run_dir / "alpha_trajectories.png"))

    return float(map50)


def run_detection_benchmark(seeds: list[int] | None = None, epochs: int = 8, lr: float = 2e-4, data_root: str = "./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Pascal VOC 2012 Detection Benchmark on {device}")
    activations = ["relu", "gelu", "swish", "prelu", "pgelu", "golu_static", "alpha_golu", "swish_adaptive"]
    seeds = seeds or [42, 123, 999, 2024, 2025]

    for act_type in activations:
        print(f"\n--- Activation: {act_type.upper()} ---")
        for seed in seeds:
            map50 = train_single_seed_detection(
                act_type,
                seed=seed,
                epochs=epochs,
                device=device,
                data_root=data_root,
                lr=lr,
                save_artifacts=True,
            )
            print(f"Seed {seed} -> VOC mAP@0.5: {map50:.4f}")


def train_and_eval(
    activation: str = "alpha_golu",
    seed: int = 42,
    epochs: int = 8,
    lr: float = 2e-4,
    data_root: str = "./data",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    save_artifacts: bool = False,
) -> float:
    """Returns VOC mAP@0.5 on a held-out Pascal VOC validation split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map50 = train_single_seed_detection(
        act_type=activation,
        seed=seed,
        epochs=epochs,
        device=device,
        data_root=data_root,
        lr=lr,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        save_artifacts=save_artifacts,
    )
    return float(map50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pascal VOC 2012 detection benchmark")
    parser.add_argument("--activation", type=str, default="alpha_golu", help="Activation to evaluate when not running the full benchmark sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Seed list for direct single-activation runs")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for direct single-activation runs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for detection training")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset cache root")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples for quick smoke runs")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional cap on evaluation samples for quick smoke runs")
    parser.add_argument("--benchmark", action="store_true", help="Run the full activation sweep benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_detection_benchmark(seeds=args.seeds, epochs=args.epochs, lr=args.lr, data_root=args.data_root)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running Pascal VOC 2012 Detection Benchmark on {device}")
        for seed in args.seeds:
            map50 = train_and_eval(
                activation=args.activation,
                seed=seed,
                epochs=args.epochs,
                data_root=args.data_root,
                lr=args.lr,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                save_artifacts=True,
            )
            print(f"Activation: {args.activation.ljust(15)} | Seed {seed} | VOC mAP@0.5: {map50:.4f}")