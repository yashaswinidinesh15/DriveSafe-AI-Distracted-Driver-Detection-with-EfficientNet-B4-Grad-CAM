"""
Data Pipeline for Distracted Driver Detection
==============================================
Handles dataset downloading, preprocessing, augmentation,
class imbalance, and DataLoader creation.

Dataset: State Farm Distracted Driver Detection (Kaggle)
Classes: 10 (c0 = safe driving, c1-c9 = distracted behaviors)
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Class Definitions
# ─────────────────────────────────────────────

CLASS_NAMES = {
    "c0": "Safe Driving",
    "c1": "Texting (Right Hand)",
    "c2": "Phone Call (Right Hand)",
    "c3": "Texting (Left Hand)",
    "c4": "Phone Call (Left Hand)",
    "c5": "Radio Adjusting",
    "c6": "Drinking",
    "c7": "Reaching Behind",
    "c8": "Hair / Makeup",
    "c9": "Talking to Passenger",
}

CLASS_TO_IDX = {cls: i for i, cls in enumerate(sorted(CLASS_NAMES.keys()))}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IDX_TO_NAME = {i: CLASS_NAMES[c] for c, i in CLASS_TO_IDX.items()}


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms with aggressive augmentation to reduce overfitting.
    Rationale:
    - RandomHorizontalFlip: mirrors real-world camera placement variation
    - RandomRotation: handles camera tilt
    - ColorJitter: handles different car interior lighting conditions
    - RandomAffine: simulates slight camera movement
    - RandomErasing: forces model to not rely on single features
    - Normalize: ImageNet statistics (pretrained backbone requirement)
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation/Test transforms - no augmentation, only resize + normalize.
    We use center crop to ensure consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Transform pipeline for single-image inference."""
    return get_val_transforms(image_size)


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────

class DistractedDriverDataset(Dataset):
    """
    PyTorch Dataset for State Farm Distracted Driver Detection.

    Supports loading from:
    1. Kaggle original structure (imgs/train/c0..c9)
    2. Pre-split directories (train/val/test)
    3. CSV manifest with image paths and labels
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        csv_file: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        self.cache: Dict = {}

        if csv_file:
            self._load_from_csv(csv_file)
        else:
            self._load_from_directory()

        logger.info(
            f"[{split.upper()}] Loaded {len(self.samples)} samples "
            f"across {len(set(self.labels))} classes"
        )

    def _load_from_directory(self):
        """Load dataset from directory structure: root/split/class_folder/image.jpg"""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            # Try flat structure: root/c0, root/c1, ...
            split_dir = self.root_dir

        self.samples = []
        self.labels = []

        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            class_name = class_folder.name
            if class_name not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in class_folder.glob("*.jpg"):
                self.samples.append(str(img_path))
                self.labels.append(label)
            for img_path in class_folder.glob("*.png"):
                self.samples.append(str(img_path))
                self.labels.append(label)

    def _load_from_csv(self, csv_file: str):
        """Load dataset from CSV with columns: image_path, label"""
        df = pd.read_csv(csv_file)
        self.samples = df["image_path"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_cache and idx in self.cache:
            image = self.cache[idx]
        else:
            img_path = self.samples[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

            if self.use_cache:
                self.cache[idx] = image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights to handle class imbalance.
        Rationale: Some behaviors (e.g., c0 safe driving) may be overrepresented.
        Weighting ensures minority classes are not ignored during training.
        """
        label_counts = Counter(self.labels)
        total = len(self.labels)
        num_classes = len(CLASS_TO_IDX)
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)
            weights.append(total / (num_classes * count))
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = torch.tensor(
            [class_weights[label] for label in self.labels], dtype=torch.float32
        )
        return sample_weights

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count per class for visualization."""
        dist = Counter(self.labels)
        return {IDX_TO_NAME[k]: v for k, v in sorted(dist.items())}


# ─────────────────────────────────────────────
# Data Splitting
# ─────────────────────────────────────────────

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, int]:
    """
    Split raw dataset into train/val/test splits with stratification.

    Stratified split ensures each class is proportionally represented
    in all splits — critical for imbalanced datasets.

    Returns dict with counts per split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    all_images = []
    all_labels = []

    for class_folder in sorted(source_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        if class_name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[class_name]
        for img in class_folder.glob("*.jpg"):
            all_images.append(img)
            all_labels.append(label)

    # Stratified split: first split off test, then split train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_images, all_labels,
        test_size=test_ratio,
        stratify=all_labels,
        random_state=random_seed
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        stratify=y_trainval,
        random_state=random_seed
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    counts = {}
    for split_name, (images, labels) in splits.items():
        split_dir = output_dir / split_name
        for class_name in CLASS_TO_IDX:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        for img_path, label in zip(images, labels):
            class_name = IDX_TO_CLASS[label]
            dest = split_dir / class_name / img_path.name
            shutil.copy2(img_path, dest)

        counts[split_name] = len(images)
        logger.info(f"Split '{split_name}': {len(images)} images")

    # Save manifest CSVs
    for split_name, (images, labels) in splits.items():
        df = pd.DataFrame({
            "image_path": [str(p) for p in images],
            "label": labels,
            "class_name": [IDX_TO_CLASS[l] for l in labels],
            "class_label": [CLASS_NAMES[IDX_TO_CLASS[l]] for l in labels],
        })
        df.to_csv(output_dir / f"{split_name}_manifest.csv", index=False)

    return counts


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    pin_memory: bool = True,
    use_cache: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders with proper sampling strategy.

    Key design decisions:
    - WeightedRandomSampler for training: compensates for class imbalance
      without artificially duplicating data (unlike oversampling).
    - No sampler for val/test: evaluate on true distribution.
    - num_workers > 0: parallel data loading is critical for GPU utilization.
    - pin_memory: speeds up CPU->GPU transfer.
    """
    data_dir = Path(data_dir)

    datasets = {
        "train": DistractedDriverDataset(
            str(data_dir), split="train",
            transform=get_train_transforms(image_size),
            use_cache=use_cache,
        ),
        "val": DistractedDriverDataset(
            str(data_dir), split="val",
            transform=get_val_transforms(image_size),
        ),
        "test": DistractedDriverDataset(
            str(data_dir), split="test",
            transform=get_val_transforms(image_size),
        ),
    }

    loaders = {}
    for split, dataset in datasets.items():
        if split == "train" and use_weighted_sampler:
            sample_weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    for split, loader in loaders.items():
        logger.info(f"[{split}] {len(loader.dataset)} samples, "
                    f"{len(loader)} batches (batch_size={batch_size})")

    return loaders


# ─────────────────────────────────────────────
# Synthetic Data Generator (for testing / CI)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(output_dir: str, samples_per_class: int = 20) -> str:
    """
    Generate a small synthetic dataset for testing pipelines without Kaggle data.
    Each 'image' is a randomly colored 224x224 PIL image.
    """
    output_dir = Path(output_dir)
    for split in ["train", "val", "test"]:
        for class_name in CLASS_TO_IDX:
            split_class_dir = output_dir / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            # Use different split sizes
            n = samples_per_class
            if split == "val":
                n = max(5, samples_per_class // 4)
            elif split == "test":
                n = max(5, samples_per_class // 4)

            for i in range(n):
                # Generate a class-specific synthetic image with patterns
                rng = np.random.RandomState(hash(f"{class_name}_{i}") % (2**31))
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)

                # Add class-specific color tint
                class_idx = CLASS_TO_IDX[class_name]
                base_color = np.array([
                    (class_idx * 25) % 255,
                    (class_idx * 50 + 100) % 255,
                    (class_idx * 75 + 50) % 255,
                ], dtype=np.uint8)

                img_array[:] = base_color
                noise = rng.randint(0, 50, (224, 224, 3), dtype=np.uint8)
                img_array = np.clip(img_array.astype(int) + noise - 25, 0, 255).astype(np.uint8)

                img = Image.fromarray(img_array)
                img.save(split_class_dir / f"{class_name}_{i:04d}.jpg")

    logger.info(f"Synthetic dataset generated at {output_dir}")
    return str(output_dir)


# ─────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────

def compute_dataset_statistics(data_dir: str) -> Dict:
    """Compute and return dataset statistics for logging/reporting."""
    data_dir = Path(data_dir)
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        counts = {}
        total = 0
        for class_name in CLASS_TO_IDX:
            class_dir = split_dir / class_name
            if class_dir.exists():
                n = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                counts[CLASS_NAMES[class_name]] = n
                total += n
        stats[split] = {"total": total, "per_class": counts}

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: generate synthetic dataset
    output = generate_synthetic_dataset("data/synthetic", samples_per_class=30)
    stats = compute_dataset_statistics(output)

    print("\n=== Dataset Statistics ===")
    for split, info in stats.items():
        print(f"\n[{split.upper()}] Total: {info['total']}")
        for cls, count in info["per_class"].items():
            print(f"  {cls:30s}: {count}")

    loaders = create_dataloaders(output, batch_size=8, num_workers=0)
    for split, loader in loaders.items():
        batch = next(iter(loader))
        images, labels = batch
        print(f"\n[{split}] Batch shape: {images.shape}, Labels: {labels[:5]}")
