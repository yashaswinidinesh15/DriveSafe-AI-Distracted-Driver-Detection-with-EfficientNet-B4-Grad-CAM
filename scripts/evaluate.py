#!/usr/bin/env python3
"""
Evaluate a trained model on the test set.
Generates confusion matrix, per-class metrics, and Grad-CAM samples.

Usage:
    python scripts/evaluate.py --model-path models/best_model.pth --data-dir data/processed
    python scripts/evaluate.py --model-path models/best_model.pth --synthetic
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate distracted driver model")
    parser.add_argument("--model-path", default="models/best_model.pth")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--architecture", default="efficientnet_b3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("evaluate")

    from src.model.architecture import create_model, load_checkpoint, ModelMetrics
    from src.data.dataset import IDX_TO_NAME
    from src.data.dataset import create_dataloaders, generate_synthetic_dataset
    from src.training.visualizations import (
        plot_confusion_matrix,
        plot_per_class_accuracy,
        generate_full_report,
    )
    import torch.nn as nn
    from torch.cuda.amp import autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    if args.synthetic:
        data_dir = generate_synthetic_dataset("data/eval_synthetic", samples_per_class=30)
    else:
        data_dir = args.data_dir

    loaders = create_dataloaders(data_dir, batch_size=args.batch_size, num_workers=0)
    test_loader = loaders["test"]

    # Model
    model = create_model(args.architecture, pretrained=False, device=device)
    model_path = Path(args.model_path)
    if model_path.exists():
        model = load_checkpoint(model, str(model_path), device)
        logger.info(f"Loaded checkpoint: {model_path}")
    else:
        logger.warning(f"Checkpoint not found at {model_path}. Using random weights.")

    model.eval()
    metrics = ModelMetrics(num_classes=10, device=device)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    logger.info(f"Evaluating on {len(test_loader.dataset)} test samples...")

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            with autocast(enabled=False):
                logits = model(images)
                loss = loss_fn(logits, targets)
            metrics.update(logits, targets)
            total_loss += loss.item()

    results = metrics.compute()
    results["test_loss"] = total_loss / len(test_loader)

    # ── Print results ──
    print("\n" + "=" * 55)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Top-1 Accuracy  : {results['accuracy_top1']:.4f} ({results['accuracy_top1']*100:.2f}%)")
    print(f"  Top-3 Accuracy  : {results['accuracy_top3']:.4f} ({results['accuracy_top3']*100:.2f}%)")
    print(f"  F1 Macro        : {results['f1_macro']:.4f}")
    print(f"  Precision Macro : {results['precision_macro']:.4f}")
    print(f"  Recall Macro    : {results['recall_macro']:.4f}")
    print(f"  AUROC           : {results['auroc']:.4f}")
    print(f"  Test Loss       : {results['test_loss']:.4f}")
    print()
    print("  Per-Class Accuracy:")
    for i, acc in enumerate(results['per_class_accuracy']):
        bar = "█" * int(acc * 20)
        print(f"    [{i}] {IDX_TO_NAME.get(i, '?'):25s} {acc:.3f}  {bar}")
    print("=" * 55)

    # ── Visualizations ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(results["confusion_matrix"])
    plot_confusion_matrix(cm, save_dir=str(output_dir))
    plot_per_class_accuracy(results["per_class_accuracy"], save_dir=str(output_dir))
    plt.close("all")

    # Save JSON results
    results_out = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results_out, f, indent=2, default=str)
    logger.info(f"Results saved to {output_dir}/test_results.json")
    logger.info(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
