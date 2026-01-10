from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    def __init__(self, transforms: Sequence[Callable], p: float = 0.5):
        self.transforms = list(transforms)
        self.p = float(p)

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        for t in self.transforms:
            x = t(x)
        return x


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = (int(size[0]), int(size[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.resize(self.size, resample=Image.BILINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.mirror(img)
        return img


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.flip(img)
        return img


class RandomRotation:
    def __init__(self, degrees: float):
        self.degrees = float(degrees)

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR)


class ColorJitter:
    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

    def _factor(self, strength: float) -> float:
        if strength <= 0:
            return 1.0
        lo = max(0.0, 1.0 - strength)
        hi = 1.0 + strength
        return random.uniform(lo, hi)

    def __call__(self, img: Image.Image) -> Image.Image:
        b = self._factor(self.brightness)
        c = self._factor(self.contrast)
        s = self._factor(self.saturation)

        # Apply in random order (common practice)
        ops = [
            ("brightness", b),
            ("contrast", c),
            ("saturation", s),
        ]
        random.shuffle(ops)
        for name, fac in ops:
            if fac == 1.0:
                continue
            if name == "brightness":
                img = ImageEnhance.Brightness(img).enhance(fac)
            elif name == "contrast":
                img = ImageEnhance.Contrast(img).enhance(fac)
            elif name == "saturation":
                img = ImageEnhance.Color(img).enhance(fac)
        return img


class RandomResizedCrop:
    """
    Simple PIL-only implementation (enough for this project; not pixel-identical to torchvision).
    """

    def __init__(self, size: int, scale: Tuple[float, float] = (0.7, 1.0), ratio: Tuple[float, float] = (0.85, 1.15)):
        self.size = int(size)
        self.scale = (float(scale[0]), float(scale[1]))
        self.ratio = (float(ratio[0]), float(ratio[1]))

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        area = w * h

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect = random.uniform(*self.ratio)
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if crop_w <= w and crop_h <= h and crop_w > 0 and crop_h > 0:
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
                img = img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
                return img.resize((self.size, self.size), resample=Image.BILINEAR)

        # Fallback to center crop + resize
        min_side = min(w, h)
        x1 = (w - min_side) // 2
        y1 = (h - min_side) // 2
        img = img.crop((x1, y1, x1 + min_side, y1 + min_side))
        return img.resize((self.size, self.size), resample=Image.BILINEAR)


class ToTensor:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0  # HWC
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return t


class Normalize:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


