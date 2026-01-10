from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data import Dataset
from PIL import Image

from .simple_transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomApply,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


@dataclass(frozen=True)
class DataConfig:
    data_root: str = "data/splits"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    use_weighted_sampler: bool = True
    use_weighted_loss: bool = True
    minority_heavy_aug: bool = True
    minority_quantile: float = 0.35  # classes <= this quantile are treated as "minority"


def build_transforms(image_size: int) -> Tuple[Callable, Callable, Callable]:
    normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    train_tf = Compose(
        [
            RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomApply([ColorJitter(0.2, 0.2, 0.2)], p=0.6),
            RandomRotation(10),
            ToTensor(),
            normalize,
        ]
    )

    heavy_train_tf = Compose(
        [
            RandomResizedCrop(image_size, scale=(0.55, 1.0)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.1),
            RandomApply([ColorJitter(0.35, 0.35, 0.35)], p=0.8),
            RandomRotation(18),
            ToTensor(),
            normalize,
        ]
    )

    eval_tf = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            normalize,
        ]
    )

    return train_tf, heavy_train_tf, eval_tf


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class FixedClassImageFolder(Dataset):
    """
    ImageFolder-like dataset with a *fixed* class_to_idx mapping.

    This prevents the classic bug where train/val/test produce different label indices
    if one split is missing a class folder.
    """

    def __init__(
        self,
        root: str | Path,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        heavy_transform: Optional[Callable] = None,
        minority_class_indices: Optional[set[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = dict(class_to_idx)
        self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]
        self.transform = transform
        self.heavy_transform = heavy_transform
        self.minority_class_indices = minority_class_indices or set()
        self.loader = self._pil_loader

        self.samples: List[Tuple[str, int]] = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), int(class_idx)))

        self.targets = [t for _, t in self.samples]

    @staticmethod
    def _pil_loader(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.heavy_transform is not None and target in self.minority_class_indices:
            sample = self.heavy_transform(sample)
        elif self.transform is not None:
            sample = self.transform(sample)

        return sample, target


def _class_counts_from_targets(targets: List[int], num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for t in targets:
        counts[int(t)] += 1
    return counts


def compute_class_weights_from_counts(counts: np.ndarray) -> torch.Tensor:
    # Inverse-frequency weights, normalized so mean weight ~= 1.
    counts = np.clip(counts.astype(np.float64), a_min=1.0, a_max=None)
    inv = 1.0 / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)


def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    root = Path(cfg.data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train split folder: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val split folder: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test split folder: {test_dir}")

    train_tf, heavy_train_tf, eval_tf = build_transforms(cfg.image_size)

    # Build class mapping from the *train* directory (authoritative)
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    # First pass to compute class counts (for sampler / class weights / minority classes)
    train_ds_base = FixedClassImageFolder(
        root=str(train_dir),
        class_to_idx=class_to_idx,
        transform=train_tf,
    )
    train_targets = list(train_ds_base.targets)
    counts = _class_counts_from_targets(train_targets, num_classes)
    q = float(cfg.minority_quantile)
    thresh = float(np.quantile(counts, q=q)) if num_classes > 0 else 0.0
    minority = {i for i, c in enumerate(counts) if float(c) <= thresh}

    if cfg.minority_heavy_aug:
        train_ds = FixedClassImageFolder(
            root=str(train_dir),
            class_to_idx=class_to_idx,
            transform=train_tf,
            heavy_transform=heavy_train_tf,
            minority_class_indices=minority,
        )
    else:
        train_ds = train_ds_base

    val_ds = FixedClassImageFolder(root=str(val_dir), class_to_idx=class_to_idx, transform=eval_tf)
    test_ds = FixedClassImageFolder(root=str(test_dir), class_to_idx=class_to_idx, transform=eval_tf)

    # Weighted sampling (oversample minority)
    sampler = None
    if cfg.use_weighted_sampler:
        class_weights = compute_class_weights_from_counts(counts).numpy()
        sample_weights = [float(class_weights[t]) for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    info = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "num_classes": num_classes,
        "train_class_counts": counts.tolist(),
        "minority_class_indices": sorted(list(minority)),
        "minority_threshold_count": float(thresh),
    }
    return train_loader, val_loader, test_loader, info


