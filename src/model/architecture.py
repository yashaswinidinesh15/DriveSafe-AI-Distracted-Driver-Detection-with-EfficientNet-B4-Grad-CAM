"""
Model Architecture for Distracted Driver Detection
====================================================
Supports multiple backbone architectures via timm library:
  - EfficientNet-B0, B3 (primary)
  - ResNet-50 (baseline comparison)
  - MobileNetV3-Large (lightweight/mobile deployment)

Design rationale:
- Transfer learning from ImageNet: dashcam images share low-level features
  (edges, textures, shapes) with natural images → faster convergence
- Custom classification head with dropout: prevents overfitting on ~22K samples
- Configurable freeze strategy: balance between feature extraction and fine-tuning
- Backbone feature hooks: required for Grad-CAM explainability
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 10


# ─────────────────────────────────────────────
# Custom Classification Head
# ─────────────────────────────────────────────

class DistractedDriverHead(nn.Module):
    """
    Custom classification head replacing the default timm head.

    Architecture:
        features → AdaptiveAvgPool → Flatten → BN → Dropout → FC → BN → Dropout → FC(10)

    Rationale for each component:
    - AdaptiveAvgPool: aggregates spatial features regardless of input size
    - BatchNorm after pooling: stabilizes training, acts as implicit regularization
    - Dropout(0.4): reduces co-adaptation between neurons, combats overfitting
    - Two-layer MLP: adds representational capacity for high-level semantic features
    - Final linear (no activation): raw logits for CrossEntropyLoss
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 512,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),  # Swish activation: smooth, non-monotonic, used in EfficientNet
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for linear layers (optimal for ReLU-family activations)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ─────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────

class DistractedDriverModel(nn.Module):
    """
    Full model: pretrained backbone + custom classification head.

    Supports:
    - Multiple backbone architectures (efficientnet_b3, resnet50, etc.)
    - Progressive unfreezing (freeze backbone → fine-tune head → unfreeze all)
    - Feature extraction for Grad-CAM
    - TorchScript export for production

    Training strategy:
    Phase 1 (epochs 1-3): Freeze backbone, train head only
        → Quickly adapts classifier to our domain without destroying pretrained features
    Phase 2 (epochs 4+): Unfreeze backbone with low LR
        → Fine-tune entire network for task-specific features
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        # Load pretrained backbone from timm
        self.backbone = timm.create_model(
            architecture,
            pretrained=pretrained,
            num_classes=0,        # Remove original classification head
            global_pool="",       # Remove global pooling (we handle in our head)
        )

        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        logger.info(f"Backbone: {architecture}, Feature dim: {self.feature_dim}")

        # Custom classification head
        self.classifier = DistractedDriverHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

        # Grad-CAM hook storage
        self._gradients = None
        self._activations = None

        # Register hooks on the last convolutional layer
        self._register_gradcam_hooks()

        self._log_model_info()

    def _get_feature_dim(self) -> int:
        """Dynamically determine output feature dimension of backbone."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            if out.dim() == 4:
                return out.shape[1]  # (B, C, H, W) → C
            return out.shape[1]  # (B, C)

    def _get_last_conv_layer(self) -> Optional[nn.Module]:
        """Find the last convolutional layer for Grad-CAM targeting."""
        last_conv = None
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_gradcam_hooks(self):
        """Register forward/backward hooks for Grad-CAM visualization."""
        last_conv = self._get_last_conv_layer()
        if last_conv is not None:
            last_conv.register_forward_hook(self._save_activations)
            last_conv.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def _log_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model: {self.architecture}")
        logger.info(f"Total params: {total:,} | Trainable: {trainable:,}")

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen. Training classifier head only.")

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None):
        """
        Unfreeze backbone parameters (Phase 2 training).
        Optionally unfreeze only the last N blocks for gradual unfreezing.
        """
        if unfreeze_last_n_blocks is None:
            # Unfreeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen.")
        else:
            # Get backbone blocks for partial unfreezing
            blocks = list(self.backbone.children())
            n = len(blocks)
            for i, block in enumerate(blocks):
                if i >= n - unfreeze_last_n_blocks:
                    for param in block.parameters():
                        param.requires_grad = True
            logger.info(f"Last {unfreeze_last_n_blocks} backbone blocks unfrozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both features and logits (for Grad-CAM and analysis)."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits

    def get_gradcam_data(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return captured gradients and activations for Grad-CAM computation."""
        return self._gradients, self._activations

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return predicted class indices and confidence scores."""
        probs = self.predict_proba(x)
        confidence, predicted = probs.max(dim=1)
        return predicted, confidence


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

SUPPORTED_ARCHITECTURES = {
    "efficientnet_b0": {"description": "Lightweight, fast inference", "params": "5.3M"},
    "efficientnet_b3": {"description": "Primary model, best accuracy/size tradeoff", "params": "12M"},
    "resnet50": {"description": "Baseline ResNet, well-understood behavior", "params": "25.6M"},
    "mobilenetv3_large_100": {"description": "Mobile-optimized, minimal latency", "params": "5.4M"},
}


def create_model(
    architecture: str = "efficientnet_b3",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout_rate: float = 0.4,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> DistractedDriverModel:
    """
    Factory function to create and optionally load a model from checkpoint.

    Args:
        architecture: backbone architecture name (see SUPPORTED_ARCHITECTURES)
        num_classes: number of output classes (10 for this dataset)
        pretrained: use ImageNet pretrained weights
        dropout_rate: dropout probability in classification head
        checkpoint_path: path to saved model checkpoint
        device: target device

    Returns:
        Initialized DistractedDriverModel
    """
    if architecture not in SUPPORTED_ARCHITECTURES:
        logger.warning(
            f"Architecture '{architecture}' not in supported list: "
            f"{list(SUPPORTED_ARCHITECTURES.keys())}. Proceeding anyway."
        )

    model = DistractedDriverModel(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )

    if checkpoint_path:
        model = load_checkpoint(model, checkpoint_path, device)

    if device:
        model = model.to(device)

    return model


def load_checkpoint(
    model: DistractedDriverModel,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> DistractedDriverModel:
    """Load model weights from checkpoint file."""
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    if "epoch" in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "best_val_acc" in checkpoint:
        logger.info(f"Best val accuracy: {checkpoint['best_val_acc']:.4f}")

    return model


def save_checkpoint(
    model: DistractedDriverModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str,
    is_best: bool = False,
):
    """Save model checkpoint with training metadata."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": model.architecture,
        "num_classes": model.num_classes,
        **metrics,
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")

    if is_best:
        import shutil
        best_path = str(save_path).replace(".pth", "_best.pth")
        shutil.copy2(save_path, best_path)
        logger.info(f"New best model saved: {best_path}")


# ─────────────────────────────────────────────
# Metrics Module
# ─────────────────────────────────────────────

class ModelMetrics:
    """
    Centralized metrics computation using torchmetrics.

    Tracked metrics:
    - Accuracy (top-1 and top-3): primary evaluation metric
    - Precision/Recall/F1 (macro): handles class imbalance in evaluation
    - Confusion Matrix: identifies systematic misclassification patterns
    - AUROC: measures discriminative ability regardless of threshold
    """

    def __init__(self, num_classes: int = NUM_CLASSES, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.num_classes = num_classes

        self.accuracy_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(self.device)
        self.accuracy_top3 = MulticlassAccuracy(num_classes=num_classes, top_k=3).to(self.device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(self.device)
        self.recall = MulticlassRecall(num_classes=num_classes, average="macro").to(self.device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(self.device)
        self.auroc = MulticlassAUROC(num_classes=num_classes, average="macro").to(self.device)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(self.device)

        self.per_class_accuracy = MulticlassAccuracy(
            num_classes=num_classes, average="none"
        ).to(self.device)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """Update all metrics with a new batch."""
        probs = F.softmax(logits, dim=1)
        self.accuracy_top1.update(logits, targets)
        self.accuracy_top3.update(logits, targets)
        self.precision.update(logits, targets)
        self.recall.update(logits, targets)
        self.f1.update(logits, targets)
        self.auroc.update(probs, targets)
        self.confusion_matrix.update(logits, targets)
        self.per_class_accuracy.update(logits, targets)

    def compute(self) -> Dict:
        """Compute and return all metrics as a dict."""
        return {
            "accuracy_top1": self.accuracy_top1.compute().item(),
            "accuracy_top3": self.accuracy_top3.compute().item(),
            "precision_macro": self.precision.compute().item(),
            "recall_macro": self.recall.compute().item(),
            "f1_macro": self.f1.compute().item(),
            "auroc": self.auroc.compute().item(),
            "confusion_matrix": self.confusion_matrix.compute().cpu().numpy().tolist(),
            "per_class_accuracy": self.per_class_accuracy.compute().cpu().numpy().tolist(),
        }

    def reset(self):
        """Reset all metric accumulators."""
        self.accuracy_top1.reset()
        self.accuracy_top3.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.confusion_matrix.reset()
        self.per_class_accuracy.reset()


# ─────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Rationale: Standard cross-entropy drives the model to output
    probability 1.0 for the true class, which can lead to overconfident
    predictions. Label smoothing distributes a small probability mass
    (epsilon) across all classes, acting as a regularizer.

    epsilon=0.1 means true class gets (1 - 0.1) = 0.9 target probability,
    and each other class gets 0.1/9 ≈ 0.011.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_loss_function(
    use_label_smoothing: bool = True,
    smoothing: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Returns the appropriate loss function.

    Design choices:
    - Label smoothing: regularization, improves calibration
    - Class weights: handles class imbalance in loss computation
    - Falls back to standard CrossEntropy if smoothing disabled
    """
    if use_label_smoothing:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test all supported architectures
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    for arch in SUPPORTED_ARCHITECTURES:
        print(f"\n{'='*50}")
        print(f"Testing: {arch}")
        model = create_model(arch, pretrained=False, device=device)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_input)
        print(f"  Output shape: {logits.shape}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test loss function
    print("\n=== Testing Loss Function ===")
    loss_fn = get_loss_function(use_label_smoothing=True)
    logits = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    loss = loss_fn(logits, targets)
    print(f"  Label smoothing loss: {loss.item():.4f}")

    # Test metrics
    print("\n=== Testing Metrics ===")
    metrics = ModelMetrics(num_classes=10, device=device)
    metrics.update(logits.to(device), targets.to(device))
    results = metrics.compute()
    print(f"  Top-1 Accuracy: {results['accuracy_top1']:.4f}")
    print(f"  F1 Macro: {results['f1_macro']:.4f}")
