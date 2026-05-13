#!/usr/bin/env python3
"""
Training Entry Point
====================
Simple CLI for launching model training.

Usage:
    # Train with defaults (uses config.yaml)
    python scripts/train.py

    # Train on synthetic data (CI/testing)
    python scripts/train.py --synthetic

    # Full training with custom settings
    python scripts/train.py \
        --data-dir data/processed \
        --architecture efficientnet_b3 \
        --epochs 30 \
        --batch-size 32 \
        --lr 0.001

    # Run ablation study after training
    python scripts/train.py --synthetic --ablation
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Distracted Driver Detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/config.yaml", help="YAML config file")
    parser.add_argument("--data-dir", default=None, help="Override data directory from config")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset (no Kaggle needed)")
    parser.add_argument("--architecture", default=None, help="Backbone architecture override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")
    parser.add_argument("--output-dir", default="models", help="Directory to save checkpoints")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study after training")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch (no ImageNet weights)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )
    logger = logging.getLogger("train")

    # ── Imports ──
    from src.training.trainer import TrainingConfig, Trainer, AblationStudy
    from src.data.dataset import create_dataloaders, generate_synthetic_dataset

    # ── Load config ──
    config = TrainingConfig.from_yaml(args.config)

    # Apply CLI overrides
    overrides = {
        "architecture": args.architecture,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dropout_rate": args.dropout,
        "output_dir": args.output_dir,
        "num_workers": args.workers,
        "pretrained": not args.no_pretrained,
    }
    if args.run_name:
        overrides["run_name"] = args.run_name
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)

    # ── Data ──
    if args.synthetic:
        logger.info("Generating synthetic dataset for training...")
        data_dir = generate_synthetic_dataset("data/synthetic_train", samples_per_class=60)
        config.data_dir = data_dir
        config.num_workers = 0
    elif args.data_dir:
        config.data_dir = args.data_dir

    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Architecture:   {config.architecture}")
    logger.info(f"Epochs:         {config.epochs}")
    logger.info(f"Batch size:     {config.batch_size}")
    logger.info(f"Learning rate:  {config.learning_rate}")

    dataloaders = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
    )

    # ── Train ──
    trainer = Trainer(config)
    best_metrics = trainer.train(dataloaders)

    # ── Summary ──
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Best Val Accuracy : {best_metrics.get('accuracy_top1', 0):.4f}")
    logger.info(f"Best Val F1       : {best_metrics.get('f1_macro', 0):.4f}")
    logger.info(f"Best Val AUROC    : {best_metrics.get('auroc', 0):.4f}")
    logger.info(f"Best Epoch        : {best_metrics.get('epoch', 0)}")
    logger.info(f"Model saved to    : {config.output_dir}/best_model.pth")

    results_path = Path(config.output_dir) / "best_metrics.json"
    with open(results_path, "w") as f:
        json.dump(best_metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to  : {results_path}")

    # ── Ablation study ──
    if args.ablation:
        logger.info("\nStarting ablation study...")
        ablation_config = TrainingConfig(config.to_dict())
        ablation_config.epochs = min(config.epochs, 10)

        ablation = AblationStudy(ablation_config, dataloaders)
        ablation.run_architecture_ablation()
        ablation.run_lr_ablation()
        ablation.run_dropout_ablation()
        ablation.save_results("mlops/ablation_results.json")
        logger.info("Ablation study saved to mlops/ablation_results.json")

    return best_metrics


if __name__ == "__main__":
    main()
