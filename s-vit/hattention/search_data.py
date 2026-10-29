"""Deterministic, class-balanced Tiny-ImageNet training/search splits."""

import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .validate_teacher import WNID


def training_splits(root, *, weight_per_class=450, architecture_per_class=50, seed=20261030):
    """Split only train images; the val/ directory is never used for search."""
    root = Path(root).resolve()
    classes = sorted((root / "wnids.txt").read_text(encoding="utf-8").splitlines())
    if len(classes) != 200 or len(set(classes)) != 200 or any(not WNID.fullmatch(x) for x in classes):
        raise ValueError("wnids.txt must contain 200 distinct WordNet IDs")
    if weight_per_class < 1 or architecture_per_class < 1:
        raise ValueError("both per-class split sizes must be positive")
    weight, architecture = [], []
    expected = weight_per_class + architecture_per_class
    for index, wnid in enumerate(classes):
        directory = root / "train" / wnid / "images"
        if not directory.is_dir():
            raise FileNotFoundError(f"training image directory missing: {directory}")
        images = sorted(p.relative_to(root).as_posix() for p in directory.iterdir()
                        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(images) != expected:
            raise ValueError(f"{wnid}: expected {expected} training images, found {len(images)}")
        random.Random(f"{seed}:{wnid}").shuffle(images)
        weight.extend((path, index) for path in images[:weight_per_class])
        architecture.extend((path, index) for path in images[weight_per_class:])
    if set(path for path, _ in weight) & set(path for path, _ in architecture):
        raise ValueError("training splits must not overlap")
    def digest(items):
        value = hashlib.sha256()
        for path, target in items:
            value.update(f"{target}\t{path}\n".encode("utf-8"))
        return value.hexdigest()
    return classes, weight, architecture, {"seed": seed,
        "weight_per_class": weight_per_class, "architecture_per_class": architecture_per_class,
        "weight_samples": len(weight), "architecture_samples": len(architecture),
        "weight_manifest_sha256": digest(weight), "architecture_manifest_sha256": digest(architecture),
        "validation_used_for_search": False}


class SearchDataset(Dataset):
    """Returns a normalized 64x64 student tensor plus the original RGB image."""

    def __init__(self, root, items):
        self.root, self.items = Path(root).resolve(), tuple(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        relative, target = self.items[index]
        with Image.open(self.root / relative) as image:
            rgb = image.convert("RGB")
        # Matches the mean/std in MPCViT's tiny_imagenet.yml; no train augmentation.
        resized = rgb.resize((64, 64), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        mean = pixels.new_tensor((0.480, 0.448, 0.397)).reshape(3, 1, 1)
        std = pixels.new_tensor((0.272, 0.265, 0.274)).reshape(3, 1, 1)
        return (pixels - mean) / std, rgb, target


def collate_search(items):
    """Keep RGB images available for the teacher's own image processor."""
    student, images, targets = zip(*items)
    return torch.stack(student), list(images), torch.tensor(targets, dtype=torch.long)
