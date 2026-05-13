"""
Training Engine for Distracted Driver Detection
================================================
Full MLOps-compliant training pipeline with:
- MLflow experiment tracking and model registry
- TensorBoard logging
- Early stopping and model checkpointing
- Mixed precision training (AMP)
- Learning rate scheduling with warmup
- Gradient clipping
- Comprehensive metric logging
- Ablation study support
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
import numpy as np
from tqdm import tqdm
import yaml

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.architecture import (
    DistractedDriverModel,
    ModelMetrics,
    create_model,
    save_checkpoint,
    load_checkpoint,
    get_loss_function,
)
from src.data.dataset import create_dataloaders, generate_synthetic_dataset, IDX_TO_NAME

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────

class TrainingConfig:
    """All hyperparameters in one place for reproducibility and logging."""

    def __init__(self, config_dict: Optional[Dict] = None):
        # Defaults
        self.architecture = "efficientnet_b3"
        self.pretrained = True
        self.num_classes = 10
        self.dropout_rate = 0.4

        self.epochs = 30
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.backbone_lr = 1e-4
        self.weight_decay = 1e-4
        self.gradient_clip_val = 1.0
        self.accumulate_grad_batches = 2
        self.label_smoothing = 0.1

        self.scheduler = "cosine_annealing"
        self.warmup_epochs = 3
        self.freeze_backbone_epochs = 3

        self.early_stopping_patience = 7
        self.image_size = 224
        self.num_workers = 4
        self.mixed_precision = True
        self.use_weighted_sampler = True

        self.data_dir = "data/processed"
        self.output_dir = "models"
        self.experiment_name = "distracted-driver-detection"
        self.run_name = None

        if config_dict:
            for k, v in config_dict.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        training_cfg = cfg.get("training", {})
        model_cfg = cfg.get("model", {})
        merged = {**training_cfg, **model_cfg}
        merged["architecture"] = model_cfg.get("architecture", "efficientnet_b3")
        return cls(merged)


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping monitors validation accuracy.
    Stops training if no improvement after `patience` epochs.
    Saves best model weights automatically.
    """

    def __init__(self, patience: int = 7, delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


# ─────────────────────────────────────────────
# Scheduler Factory
# ─────────────────────────────────────────────

def build_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """
    Build learning rate scheduler with optional warmup.

    Strategy:
    - Linear warmup for first N epochs: prevents large initial updates
      that can destabilize pretrained weights
    - CosineAnnealing: smoothly decays LR to minimum, then repeats
      → better than step decay for convergence to flat minima
    - OneCycleLR: alternative aggressive scheduler for fast training
    """
    if config.scheduler == "cosine_annealing":
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=1e-6,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs],
        )

    elif config.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    else:
        return CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)


# ─────────────────────────────────────────────
# Single Epoch Train / Eval
# ─────────────────────────────────────────────

def train_one_epoch(
    model: DistractedDriverModel,
    loader,
    optimizer,
    loss_fn,
    scaler: GradScaler,
    config: TrainingConfig,
    epoch: int,
    writer: SummaryWriter,
    device: torch.device,
) -> Dict:
    """Run one training epoch with AMP, gradient accumulation, and clipping."""
    model.train()
    metrics = ModelMetrics(num_classes=config.num_classes, device=device)

    total_loss = 0.0
    num_batches = len(loader)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast(enabled=config.mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, targets)
            # Scale loss for gradient accumulation
            loss = loss / config.accumulate_grad_batches

        scaler.scale(loss).backward()

        # Gradient accumulation: update only every N batches
        if (batch_idx + 1) % config.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_val
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            metrics.update(logits.detach(), targets)

        total_loss += loss.item() * config.accumulate_grad_batches
        pbar.set_postfix({"loss": f"{loss.item() * config.accumulate_grad_batches:.4f}"})

        # Log batch-level metrics every 50 steps
        global_step = epoch * num_batches + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar("Batch/train_loss", loss.item(), global_step)

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = total_loss / num_batches

    return epoch_metrics


@torch.no_grad()
def evaluate(
    model: DistractedDriverModel,
    loader,
    loss_fn,
    config: TrainingConfig,
    device: torch.device,
    split: str = "val",
) -> Dict:
    """Evaluate model on val or test split."""
    model.eval()
    metrics = ModelMetrics(num_classes=config.num_classes, device=device)
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"  [{split.upper()}]", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=config.mixed_precision):
            logits = model(images)
            loss = loss_fn(logits, targets)

        metrics.update(logits, targets)
        total_loss += loss.item()

    epoch_metrics = metrics.compute()
    epoch_metrics["loss"] = total_loss / len(loader)
    return epoch_metrics


# ─────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────

class Trainer:
    """
    MLOps-compliant trainer with full experiment tracking.

    Features:
    - MLflow: hyperparameter logging, metric tracking, model registry
    - TensorBoard: real-time training visualization
    - Checkpointing: best model + periodic saves
    - Two-phase training: frozen → unfrozen backbone
    - Comprehensive per-epoch and per-class metrics
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        tb_dir = Path("runs") / config.experiment_name / (config.run_name or "default")
        self.writer = SummaryWriter(str(tb_dir))
        logger.info(f"TensorBoard logs: {tb_dir}")

        # Setup MLflow
        mlflow.set_experiment(config.experiment_name)

    def _build_optimizer(self, model: DistractedDriverModel, phase: int = 1):
        """
        Build optimizer with parameter groups.

        Phase 1 (backbone frozen): only train classifier head
        Phase 2 (unfrozen): backbone gets lower LR (1/10), head gets full LR
        This prevents the pretrained backbone from adapting too fast and
        losing valuable ImageNet representations.
        """
        if phase == 1:
            params = [p for p in model.classifier.parameters() if p.requires_grad]
            return AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            return AdamW([
                {"params": model.backbone.parameters(), "lr": self.config.backbone_lr},
                {"params": model.classifier.parameters(), "lr": self.config.learning_rate},
            ], weight_decay=self.config.weight_decay)

    def _log_epoch_metrics(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch: int,
        lr: float,
        run,
    ):
        """Log metrics to TensorBoard and MLflow."""
        # TensorBoard
        for key, val in train_metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(f"Train/{key}", val, epoch)
        for key, val in val_metrics.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(f"Val/{key}", val, epoch)
        self.writer.add_scalar("LR", lr, epoch)

        # MLflow
        mlflow.log_metrics({
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy_top1"],
            "train_f1": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy_top1"],
            "val_f1": val_metrics["f1_macro"],
            "val_auroc": val_metrics["auroc"],
            "learning_rate": lr,
        }, step=epoch)

    def train(self, dataloaders: Dict) -> Dict:
        """
        Full training loop.

        Returns best validation metrics.
        """
        config = self.config
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        # Build model
        model = create_model(
            architecture=config.architecture,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate,
            device=self.device,
        )

        # Loss function
        class_weights = None
        if hasattr(train_loader.dataset, "get_class_weights"):
            class_weights = train_loader.dataset.get_class_weights().to(self.device)
        loss_fn = get_loss_function(
            use_label_smoothing=True,
            smoothing=config.label_smoothing,
            class_weights=class_weights,
        )

        # Phase 1: freeze backbone
        model.freeze_backbone()
        optimizer = self._build_optimizer(model, phase=1)
        scheduler = build_scheduler(optimizer, config, len(train_loader))
        scaler = GradScaler(enabled=config.mixed_precision)
        early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        best_val_acc = 0.0
        best_metrics = {}
        history = []

        with mlflow.start_run(run_name=config.run_name) as run:
            # Log hyperparameters
            mlflow.log_params(config.to_dict())
            mlflow.log_param("device", str(self.device))
            mlflow.log_param("num_train_samples", len(train_loader.dataset))
            mlflow.log_param("num_val_samples", len(val_loader.dataset))

            logger.info(f"Starting training: {config.epochs} epochs, device={self.device}")
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            for epoch in range(config.epochs):
                epoch_start = time.time()

                # Phase transition: unfreeze backbone after warmup
                if epoch == config.freeze_backbone_epochs:
                    logger.info("=== Phase 2: Unfreezing backbone ===")
                    model.unfreeze_backbone()
                    optimizer = self._build_optimizer(model, phase=2)
                    scheduler = build_scheduler(optimizer, config, len(train_loader))
                    scaler = GradScaler(enabled=config.mixed_precision)

                # Train
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, scaler,
                    config, epoch, self.writer, self.device
                )

                # Validate
                val_metrics = evaluate(
                    model, val_loader, loss_fn, config, self.device, "val"
                )

                # Step scheduler
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # Log
                self._log_epoch_metrics(train_metrics, val_metrics, epoch, current_lr, run)

                epoch_time = time.time() - epoch_start
                val_acc = val_metrics["accuracy_top1"]

                logger.info(
                    f"Epoch [{epoch+1:3d}/{config.epochs}] "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Acc: {train_metrics['accuracy_top1']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Acc: {val_acc:.4f} "
                    f"F1: {val_metrics['f1_macro']:.4f} "
                    f"AUROC: {val_metrics['auroc']:.4f} "
                    f"LR: {current_lr:.6f} "
                    f"[{epoch_time:.1f}s]"
                )

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = {**val_metrics, "epoch": epoch + 1}
                    save_checkpoint(
                        model, optimizer, epoch,
                        {"best_val_acc": best_val_acc, "val_f1": val_metrics["f1_macro"]},
                        self.output_dir / "best_model.pth",
                        is_best=True,
                    )
                    mlflow.pytorch.log_model(model, "best_model")
                    logger.info(f"  ✓ New best model (val_acc={best_val_acc:.4f})")

                # Periodic checkpoint
                if (epoch + 1) % 5 == 0:
                    save_checkpoint(
                        model, optimizer, epoch,
                        {"val_acc": val_acc},
                        self.output_dir / f"checkpoint_epoch_{epoch+1}.pth",
                    )

                history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["accuracy_top1"],
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_acc,
                    "val_f1": val_metrics["f1_macro"],
                    "lr": current_lr,
                })

                # Early stopping
                if early_stopping(val_acc):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Save training history
            history_path = self.output_dir / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            mlflow.log_artifact(str(history_path))

            # Log summary metrics
            mlflow.log_metrics({
                "best_val_accuracy": best_val_acc,
                "best_val_f1": best_metrics.get("f1_macro", 0),
                "best_val_auroc": best_metrics.get("auroc", 0),
                "best_epoch": best_metrics.get("epoch", 0),
            })

        self.writer.close()
        logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
        return best_metrics


# ─────────────────────────────────────────────
# Ablation Study Engine
# ─────────────────────────────────────────────

class AblationStudy:
    """
    Systematic ablation study across hyperparameters and architectures.

    Studies conducted:
    1. Architecture comparison: EfficientNet-B0/B3 vs ResNet50 vs MobileNetV3
    2. Learning rate sweep: 1e-4, 1e-3, 1e-2
    3. Dropout rate: 0.2, 0.4, 0.5
    4. Data augmentation: with vs without
    5. Freeze strategy: full, partial, none
    """

    def __init__(self, base_config: TrainingConfig, dataloaders: Dict):
        self.base_config = base_config
        self.dataloaders = dataloaders
        self.results = []

    def run_architecture_ablation(self) -> List[Dict]:
        """Compare different backbone architectures."""
        architectures = [
            "efficientnet_b0",
            "efficientnet_b3",
            "resnet50",
            "mobilenetv3_large_100",
        ]
        return self._run_sweep("architecture", architectures, "architecture")

    def run_lr_ablation(self) -> List[Dict]:
        """Sweep learning rates."""
        lrs = [1e-4, 1e-3, 1e-2]
        return self._run_sweep("learning_rate_sweep", lrs, "learning_rate")

    def run_dropout_ablation(self) -> List[Dict]:
        """Sweep dropout rates."""
        dropouts = [0.2, 0.4, 0.5]
        return self._run_sweep("dropout_sweep", dropouts, "dropout_rate")

    def _run_sweep(self, sweep_name: str, values: List, param_name: str) -> List[Dict]:
        """Generic sweep runner."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: {sweep_name}")
        logger.info(f"  Sweeping {param_name} over: {values}")
        logger.info('='*60)

        results = []
        for value in values:
            config_dict = self.base_config.to_dict()
            config_dict[param_name] = value
            config_dict["epochs"] = min(self.base_config.epochs, 10)  # Short ablation runs
            config_dict["run_name"] = f"{sweep_name}_{param_name}={value}"
            config_dict["experiment_name"] = f"ablation_{sweep_name}"

            config = TrainingConfig(config_dict)
            trainer = Trainer(config)
            best = trainer.train(self.dataloaders)

            result = {
                "sweep": sweep_name,
                param_name: value,
                "best_val_accuracy": best.get("accuracy_top1", 0),
                "best_val_f1": best.get("f1_macro", 0),
                "best_epoch": best.get("epoch", 0),
            }
            results.append(result)
            self.results.append(result)
            logger.info(f"  {param_name}={value}: val_acc={result['best_val_accuracy']:.4f}")

        return results

    def save_results(self, output_path: str):
        """Save all ablation results to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Ablation results saved to {output_path}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def train_from_config(config_path: str, data_dir: Optional[str] = None) -> Dict:
    """Main training entry point from YAML config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler(),
        ],
    )

    config = TrainingConfig.from_yaml(config_path)
    if data_dir:
        config.data_dir = data_dir

    logger.info("=== Distracted Driver Detection - Training ===")
    logger.info(f"Config: {config.to_dict()}")

    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
    )

    # Train
    trainer = Trainer(config)
    best_metrics = trainer.train(dataloaders)

    logger.info(f"Best metrics: {best_metrics}")
    return best_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: train on synthetic dataset
    logger.info("Generating synthetic dataset for demo training...")
    synthetic_dir = generate_synthetic_dataset("data/synthetic_demo", samples_per_class=40)

    config = TrainingConfig({
        "architecture": "efficientnet_b0",
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "freeze_backbone_epochs": 1,
        "early_stopping_patience": 3,
        "data_dir": synthetic_dir,
        "output_dir": "models/demo",
        "num_workers": 0,
        "mixed_precision": False,
        "pretrained": False,
        "run_name": "synthetic_demo",
    })

    dataloaders = create_dataloaders(
        data_dir=synthetic_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=0,
    )

    trainer = Trainer(config)
    best_metrics = trainer.train(dataloaders)
    print(f"\nBest val accuracy: {best_metrics.get('accuracy_top1', 0):.4f}")
