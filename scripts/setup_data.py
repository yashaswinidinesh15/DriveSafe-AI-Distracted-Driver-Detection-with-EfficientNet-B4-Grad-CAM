#!/usr/bin/env python3
"""
Dataset Setup Script
====================
Downloads the State Farm Distracted Driver Detection dataset
from Kaggle and prepares it for training.

Usage:
    python scripts/setup_data.py --kaggle          # Download from Kaggle
    python scripts/setup_data.py --synthetic       # Use synthetic data (for testing)
    python scripts/setup_data.py --synthetic --n 100  # 100 samples per class
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def setup_kaggle(output_dir: str = "data"):
    """Download State Farm dataset from Kaggle."""
    import subprocess
    import zipfile

    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading State Farm Distracted Driver Detection dataset...")
    logger.info("Make sure your kaggle.json is at ~/.kaggle/kaggle.json")

    result = subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "state-farm-distracted-driver-detection",
            "-p", str(raw_dir),
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error(f"Kaggle download failed:\n{result.stderr}")
        logger.info("\nTo set up Kaggle:")
        logger.info("1. Go to https://www.kaggle.com/settings")
        logger.info("2. Click 'Create New Token' under API")
        logger.info("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        logger.info("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    logger.info("Download complete. Extracting...")
    for zipfile_path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(zipfile_path, "r") as zf:
            zf.extractall(raw_dir)
        zipfile_path.unlink()
        logger.info(f"Extracted: {zipfile_path.name}")

    return True


def setup_synthetic(output_dir: str = "data", n_per_class: int = 50):
    """Generate synthetic dataset for testing."""
    from src.data.dataset import generate_synthetic_dataset

    logger.info(f"Generating synthetic dataset ({n_per_class} samples/class)...")
    data_dir = generate_synthetic_dataset(str(Path(output_dir) / "synthetic"), n_per_class)
    logger.info(f"Synthetic dataset ready at: {data_dir}")
    return data_dir


def split_and_process(raw_dir: str, processed_dir: str = "data/processed"):
    """Split raw dataset into train/val/test."""
    from src.data.dataset import split_dataset, compute_dataset_statistics

    raw_path = Path(raw_dir)
    train_src = raw_path / "imgs" / "train"
    if not train_src.exists():
        train_src = raw_path

    logger.info(f"Splitting dataset from {train_src}...")
    counts = split_dataset(
        source_dir=str(train_src),
        output_dir=processed_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    stats = compute_dataset_statistics(processed_dir)

    logger.info("\n=== Dataset Statistics ===")
    for split, info in stats.items():
        logger.info(f"\n[{split.upper()}] Total: {info['total']}")
        for cls, count in info["per_class"].items():
            logger.info(f"  {cls:30s}: {count}")

    return processed_dir


def main():
    parser = argparse.ArgumentParser(description="Setup distracted driver dataset")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--n", type=int, default=50, help="Samples per class (synthetic)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--skip-split", action="store_true", help="Skip train/val/test split")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.synthetic:
        data_dir = setup_synthetic(args.data_dir, args.n)
        logger.info(f"\n✅ Synthetic data ready at: {data_dir}")
        logger.info("Run training with: python src/training/trainer.py")

    elif args.kaggle:
        success = setup_kaggle(args.data_dir)
        if success and not args.skip_split:
            processed_dir = split_and_process(
                str(Path(args.data_dir) / "raw"),
                str(Path(args.data_dir) / "processed"),
            )
            logger.info(f"\n✅ Processed data ready at: {processed_dir}")
            logger.info("Run training with: python mlops/pipelines/pipeline.py --config configs/config.yaml")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
