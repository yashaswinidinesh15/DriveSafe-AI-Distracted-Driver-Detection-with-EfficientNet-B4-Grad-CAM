"""
Grad-CAM Explainability for Distracted Driver Detection
=========================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize WHICH regions of dashcam images the model focuses on
when making predictions.

Why Grad-CAM?
- Model transparency: shows if model looks at correct features
  (e.g., hands/phone for texting class, not car dashboard)
- Debugging: identifies when model uses spurious correlations
- Trust: stakeholders (insurers, fleet managers) need to trust predictions
- Regulatory: explainability is increasingly required for safety-critical AI

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Grad-CAM Core
# ─────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Map generator.

    Algorithm:
    1. Forward pass: get activations at target layer
    2. Backward pass: compute gradients of class score w.r.t. activations
    3. Global average pool gradients → weights (importance per channel)
    4. Weighted sum of activation maps → raw CAM
    5. ReLU: keep only positive contributions
    6. Upsample to input size and normalize

    Intuition: channels with large gradient magnitude contribute most
    to the predicted class → their activation maps are most "important"
    """

    def __init__(self, model, target_layer_name: Optional[str] = None):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []

        # Find target layer
        self.target_layer = self._find_target_layer(target_layer_name)
        self._register_hooks()

    def _find_target_layer(self, layer_name: Optional[str]):
        """Find the last convolutional layer or specified layer."""
        if layer_name:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    return module

        # Default: last Conv2d in backbone
        last_conv = None
        for module in self.model.backbone.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks."""
        if self.target_layer is None:
            logger.warning("No target layer found for Grad-CAM")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def remove_hooks(self):
        """Clean up hooks to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, C, H, W]
            target_class: class to explain (None = use predicted class)

        Returns:
            heatmap: normalized CAM [H, W] in [0, 1]
            target_class: class that was explained
            confidence: model confidence for target class
        """
        self.model.eval()

        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM hooks did not capture data")
            return np.zeros((224, 224)), target_class, confidence

        # Global average pooling of gradients → channel weights
        # Shape: [C, H, W] → [C]
        weights = self.gradients[0].mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        # Shape: [C, H, W] → [H, W]
        cam = (weights[:, None, None] * self.activations[0]).sum(dim=0)

        # ReLU: only positive contributions matter
        cam = F.relu(torch.tensor(cam)).numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class, confidence

    def __del__(self):
        self.remove_hooks()


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def overlay_gradcam(
    image: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: original image (PIL or numpy BGR)
        cam: Grad-CAM array [H, W] in [0, 1]
        alpha: heatmap blend factor (0=no overlay, 1=full heatmap)
        colormap: OpenCV colormap for heatmap

    Returns:
        blended image as numpy array [H, W, 3] uint8
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    h, w = image_np.shape[:2]

    # Resize CAM to image size using bilinear interpolation
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to uint8 heatmap
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlaid = (1 - alpha) * image_np + alpha * heatmap_rgb
    return np.uint8(overlaid)


def create_gradcam_figure(
    original_image: Image.Image,
    cam: np.ndarray,
    class_name: str,
    confidence: float,
    all_probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive Grad-CAM visualization figure.

    Shows:
    1. Original image
    2. Grad-CAM heatmap only
    3. Overlaid image with heatmap
    4. Class probability bar chart (if probs provided)
    """
    ncols = 4 if all_probs is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    image_np = np.array(original_image.convert("RGB"))
    h, w = image_np.shape[:2]

    # 1. Original
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Heatmap only
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    im = axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    overlay = overlay_gradcam(original_image, cam, alpha=0.5)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nPrediction: {class_name}\nConfidence: {confidence:.1%}",
        fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    # 4. Class probabilities
    if all_probs is not None and class_names is not None:
        y_pos = np.arange(len(class_names))
        colors = ["#e74c3c" if i == np.argmax(all_probs) else "#3498db"
                  for i in range(len(class_names))]
        bars = axes[3].barh(y_pos, all_probs * 100, color=colors, edgecolor="white")
        axes[3].set_yticks(y_pos)
        axes[3].set_yticklabels(class_names, fontsize=9)
        axes[3].set_xlabel("Confidence (%)", fontsize=10)
        axes[3].set_title("Class Probabilities", fontsize=12, fontweight="bold")
        axes[3].set_xlim(0, 100)

        # Add value labels
        for bar, prob in zip(bars, all_probs):
            if prob > 0.02:
                axes[3].text(
                    bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", ha="left", fontsize=8
                )

    plt.suptitle(
        f"Distracted Driver Detection — Grad-CAM Explainability",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Grad-CAM figure saved: {save_path}")

    return fig


# ─────────────────────────────────────────────
# Inference + Grad-CAM Pipeline
# ─────────────────────────────────────────────

class ExplainablePredictor:
    """
    Combined inference + Grad-CAM for production use.

    Provides:
    - Class prediction with confidence scores
    - Grad-CAM heatmap for predicted class
    - Multi-class Grad-CAM for top-K classes
    - JSON-serializable output for API responses
    """

    IDX_TO_NAME = {
        0: "Safe Driving",
        1: "Texting (Right Hand)",
        2: "Phone Call (Right Hand)",
        3: "Texting (Left Hand)",
        4: "Phone Call (Left Hand)",
        5: "Radio Adjusting",
        6: "Drinking",
        7: "Reaching Behind",
        8: "Hair / Makeup",
        9: "Talking to Passenger",
    }

    def __init__(self, model, device: Optional[torch.device] = None, image_size: int = 224):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.gradcam = GradCAM(model)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
        generate_cam: bool = True,
    ) -> Dict:
        """
        Full prediction pipeline.

        Returns dict with:
        - predicted_class: int
        - predicted_label: str
        - confidence: float
        - top_k_predictions: List[{class, label, confidence}]
        - cam: np.ndarray (optional)
        - cam_overlay: np.ndarray (optional)
        """
        # Load image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # Top-K predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_preds = [
            {
                "class_idx": int(i),
                "label": self.IDX_TO_NAME[int(i)],
                "confidence": float(probs[i]),
            }
            for i in top_k_indices
        ]

        result = {
            "predicted_class": predicted_class,
            "predicted_label": self.IDX_TO_NAME[predicted_class],
            "confidence": confidence,
            "all_probabilities": probs.tolist(),
            "top_k_predictions": top_k_preds,
            "is_distracted": predicted_class != 0,
        }

        # Grad-CAM
        if generate_cam:
            input_tensor_grad = self.transform(pil_image).unsqueeze(0).to(self.device)
            input_tensor_grad.requires_grad_(True)
            cam, _, _ = self.gradcam.generate(input_tensor_grad, target_class=predicted_class)
            cam_overlay = overlay_gradcam(pil_image, cam, alpha=0.45)

            result["cam"] = cam
            result["cam_overlay"] = cam_overlay
            result["pil_image"] = pil_image

        return result

    def predict_batch(
        self,
        images: List[Union[str, Image.Image]],
        batch_size: int = 16,
    ) -> List[Dict]:
        """Batch inference without Grad-CAM (for speed)."""
        results = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            tensors = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    pil = Image.open(str(img)).convert("RGB")
                else:
                    pil = img.convert("RGB")
                tensors.append(self.transform(pil))

            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()

            for j, prob in enumerate(probs):
                pred_class = int(np.argmax(prob))
                results.append({
                    "predicted_class": pred_class,
                    "predicted_label": self.IDX_TO_NAME[pred_class],
                    "confidence": float(prob[pred_class]),
                    "is_distracted": pred_class != 0,
                    "all_probabilities": prob.tolist(),
                })

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic model
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.model.architecture import create_model

    device = torch.device("cpu")
    model = create_model("efficientnet_b0", pretrained=False, device=device)

    predictor = ExplainablePredictor(model, device=device)

    # Test with a synthetic image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = predictor.predict(dummy_img, generate_cam=True)

    print(f"\nPrediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Is distracted: {result['is_distracted']}")
    print(f"Top-3:")
    for p in result["top_k_predictions"]:
        print(f"  {p['label']:30s}: {p['confidence']:.1%}")
    print(f"CAM shape: {result['cam'].shape}")
    print(f"Overlay shape: {result['cam_overlay'].shape}")
