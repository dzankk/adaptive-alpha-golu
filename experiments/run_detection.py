"""
Benchmark: Pascal VOC 2012 Object Detection
===========================================
Trains a lightweight anchor-free detector on real Pascal VOC annotations and
reports VOC-style mAP@0.5 on a held-out validation split.

This runner is intentionally grounded in a real benchmark rather than a toy or
proxy dataset.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VOCDetection
from torchvision.ops import box_iou, nms
from torchvision.transforms import functional as TF

from models.alpha_golu import AlphaGoLU as AdaptiveAlphaGoLU, StaticGoLU


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
        init_val = float(init_alpha)
        init_raw = math.log(math.expm1(init_val)) if init_val < 20 else init_val
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha)

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


def get_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
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


class DetectionBackbone(nn.Module):
    """Lightweight feature extractor with selectable activations."""

    def __init__(self, act_type: str = "relu"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            get_activation(act_type),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            get_activation(act_type),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            get_activation(act_type),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            get_activation(act_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class DenseVOCDetector(nn.Module):
    """Dense anchor-free detector trained on Pascal VOC."""

    def __init__(self, act_type: str = "relu", num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = DetectionBackbone(act_type=act_type)
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            get_activation(act_type),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            get_activation(act_type),
            nn.Conv2d(128, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.cls_head(features), self.box_head(features)


def build_dense_targets(
    targets: List[Dict[str, torch.Tensor]],
    feat_h: int,
    feat_w: int,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(targets)
    cls_targets = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.long, device=device)
    box_targets = torch.zeros((batch_size, feat_h, feat_w, 4), dtype=torch.float32, device=device)
    positive_mask = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.bool, device=device)
    assigned_area = torch.zeros((batch_size, feat_h, feat_w), dtype=torch.float32, device=device)

    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device)
        if boxes.numel() == 0:
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        sizes = boxes[:, 2:] - boxes[:, :2]
        x_cells = torch.clamp((centers[:, 0] / image_size * feat_w).long(), min=0, max=feat_w - 1)
        y_cells = torch.clamp((centers[:, 1] / image_size * feat_h).long(), min=0, max=feat_h - 1)
        areas = sizes[:, 0] * sizes[:, 1]

        for obj_index in range(boxes.size(0)):
            gx = int(x_cells[obj_index].item())
            gy = int(y_cells[obj_index].item())
            area = float(areas[obj_index].item())

            if (not positive_mask[batch_index, gy, gx]) or area > float(assigned_area[batch_index, gy, gx].item()):
                cls_targets[batch_index, gy, gx] = labels[obj_index]
                cx = centers[obj_index, 0] / image_size
                cy = centers[obj_index, 1] / image_size
                w = sizes[obj_index, 0] / image_size
                h = sizes[obj_index, 1] / image_size
                box_targets[batch_index, gy, gx] = torch.tensor([cx, cy, w, h], device=device)
                positive_mask[batch_index, gy, gx] = True
                assigned_area[batch_index, gy, gx] = area

    return cls_targets, box_targets, positive_mask


def decode_predictions(
    cls_logits: torch.Tensor,
    box_logits: torch.Tensor,
    image_size: int,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(cls_logits, dim=0)  # [C, H, W]
    class_scores, class_labels = probs[1:].max(dim=0)
    class_labels = class_labels + 1

    box_params = torch.stack(
        [
            torch.sigmoid(box_logits[0]),
            torch.sigmoid(box_logits[1]),
            F.softplus(box_logits[2]),
            F.softplus(box_logits[3]),
        ],
        dim=0,
    )

    class_scores = class_scores.reshape(-1)
    class_labels = class_labels.reshape(-1)
    box_params = box_params.permute(1, 2, 0).reshape(-1, 4)

    keep = class_scores > score_thresh
    if keep.sum() == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=cls_logits.device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=cls_logits.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=cls_logits.device),
        }

    class_scores = class_scores[keep]
    class_labels = class_labels[keep]
    box_params = box_params[keep]

    centers = box_params[:, :2] * image_size
    sizes = box_params[:, 2:] * image_size
    boxes = torch.cat([centers - sizes / 2.0, centers + sizes / 2.0], dim=1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0.0, float(image_size))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(image_size))

    final_boxes = []
    final_scores = []
    final_labels = []
    for label in class_labels.unique(sorted=True):
        label_mask = class_labels == label
        label_boxes = boxes[label_mask]
        label_scores = class_scores[label_mask]
        if label_boxes.numel() == 0:
            continue

        keep_indices = nms(label_boxes, label_scores, nms_thresh)
        final_boxes.append(label_boxes[keep_indices])
        final_scores.append(label_scores[keep_indices])
        final_labels.append(torch.full((len(keep_indices),), int(label.item()), dtype=torch.long, device=cls_logits.device))

    if not final_boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=cls_logits.device),
            "scores": torch.zeros((0,), dtype=torch.float32, device=cls_logits.device),
            "labels": torch.zeros((0,), dtype=torch.long, device=cls_logits.device),
        }

    return {
        "boxes": torch.cat(final_boxes, dim=0),
        "scores": torch.cat(final_scores, dim=0),
        "labels": torch.cat(final_labels, dim=0),
    }


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
    image_size: int,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
) -> float:
    model.eval()

    detections_by_class: Dict[int, List[Tuple[float, int, torch.Tensor]]] = defaultdict(list)
    ground_truth_by_class: Dict[int, Dict[int, List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in dataloader:
            batch_images = torch.stack([image.to(device) for image in images])
            cls_logits, box_logits = model(batch_images)

            for batch_index, target in enumerate(targets):
                image_id = int(target["image_id"].item())
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    ground_truth_by_class[int(gt_label.item())][image_id].append(gt_box.clone())

                prediction = decode_predictions(
                    cls_logits[batch_index],
                    box_logits[batch_index],
                    image_size=image_size,
                    score_thresh=score_thresh,
                    nms_thresh=nms_thresh,
                )

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
    image_size: int = 320,
    train_split_ratio: float = 0.9,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> float:
    set_seed(seed)

    full_dataset = PascalVOCDataset(
        root="./data",
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

    model = DenseVOCDetector(act_type=act_type, num_classes=len(VOC_CLASSES) + 1).to(device)
    optimizer = get_optimizer(model, lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            batch_images = torch.stack([image.to(device) for image in images])
            cls_logits, box_logits = model(batch_images)

            _, _, feat_h, feat_w = cls_logits.shape
            cls_targets, box_targets, positive_mask = build_dense_targets(
                targets,
                feat_h=feat_h,
                feat_w=feat_w,
                image_size=image_size,
                device=device,
            )

            cls_loss = F.cross_entropy(
                cls_logits.permute(0, 2, 3, 1).reshape(-1, len(VOC_CLASSES) + 1),
                cls_targets.reshape(-1),
            )

            pred_box = torch.stack(
                [
                    torch.sigmoid(box_logits[:, 0]),
                    torch.sigmoid(box_logits[:, 1]),
                    F.softplus(box_logits[:, 2]),
                    F.softplus(box_logits[:, 3]),
                ],
                dim=1,
            ).permute(0, 2, 3, 1)

            if positive_mask.any():
                box_loss = F.smooth_l1_loss(pred_box[positive_mask], box_targets[positive_mask])
            else:
                box_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + 2.0 * box_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    map50 = evaluate_map50(model, eval_loader, device=device, image_size=image_size)
    return float(map50)


def run_detection_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Pascal VOC 2012 Detection Benchmark on {device}")
    activations = ["relu", "gelu", "swish", "prelu", "pgelu", "golu_static", "alpha_golu", "swish_adaptive"]

    for act_type in activations:
        map50 = train_single_seed_detection(act_type, seed=42, epochs=3, device=device)
        print(f"Activation: {act_type.ljust(15)} | VOC mAP@0.5: {map50:.4f}")


def train_and_eval(activation: str = "alpha_golu", seed: int = 42, epochs: int = 3) -> float:
    """Returns VOC mAP@0.5 on a held-out Pascal VOC validation split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map50 = train_single_seed_detection(act_type=activation, seed=seed, epochs=epochs, device=device)
    return float(map50)


if __name__ == "__main__":
    run_detection_benchmark()