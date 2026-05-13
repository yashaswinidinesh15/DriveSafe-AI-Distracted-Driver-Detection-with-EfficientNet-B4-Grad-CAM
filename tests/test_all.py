"""
Test Suite — Distracted Driver Detection
=========================================
Covers:
- Dataset loading and transforms
- Model architecture and forward pass
- Loss functions and metrics
- Grad-CAM generation
- API endpoint validation
- Training loop (minimal)
- Pipeline step validation
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# Data Tests
# ─────────────────────────────────────────────

class TestDataset(unittest.TestCase):

    def setUp(self):
        """Create a tiny synthetic dataset for testing."""
        from src.data.dataset import generate_synthetic_dataset
        self.tmp_dir = tempfile.mkdtemp()
        self.data_dir = generate_synthetic_dataset(self.tmp_dir, samples_per_class=8)

    def test_dataset_loads(self):
        """Dataset should load all samples from directory."""
        from src.data.dataset import DistractedDriverDataset, get_val_transforms

        dataset = DistractedDriverDataset(
            self.data_dir, split="train",
            transform=get_val_transforms(224),
        )
        self.assertGreater(len(dataset), 0)

    def test_dataset_returns_correct_shape(self):
        """Each sample should be (C=3, H=224, W=224) tensor and int label."""
        from src.data.dataset import DistractedDriverDataset, get_val_transforms

        dataset = DistractedDriverDataset(
            self.data_dir, split="train",
            transform=get_val_transforms(224),
        )
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIsInstance(label, int)
        self.assertIn(label, range(10))

    def test_class_weights_computed(self):
        """Class weights should be positive tensors."""
        from src.data.dataset import DistractedDriverDataset, get_val_transforms

        dataset = DistractedDriverDataset(
            self.data_dir, split="train",
            transform=get_val_transforms(224),
        )
        weights = dataset.get_class_weights()
        self.assertEqual(len(weights), 10)
        self.assertTrue((weights > 0).all())

    def test_dataloader_creates(self):
        """DataLoaders should be created for all splits."""
        from src.data.dataset import create_dataloaders

        loaders = create_dataloaders(
            self.data_dir, batch_size=4, num_workers=0
        )
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

    def test_train_transforms_augment(self):
        """Train transforms should return normalized tensor."""
        from src.data.dataset import get_train_transforms

        transform = get_train_transforms(224)
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = transform(img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        # Normalized tensors should have values outside [0,1]
        self.assertTrue(tensor.min() < 0 or tensor.max() > 1)

    def test_class_distribution(self):
        """Class distribution dict should have 10 entries."""
        from src.data.dataset import DistractedDriverDataset, get_val_transforms

        dataset = DistractedDriverDataset(
            self.data_dir, split="train",
            transform=get_val_transforms(224),
        )
        dist = dataset.get_class_distribution()
        self.assertEqual(len(dist), 10)


# ─────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────

class TestModel(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, 3, 224, 224)

    def test_efficientnet_b0_forward(self):
        """EfficientNet-B0 should produce (B, 10) output."""
        from src.model.architecture import create_model

        model = create_model("efficientnet_b0", pretrained=False, device=self.device)
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 10))

    def test_resnet50_forward(self):
        """ResNet-50 should produce (B, 10) output."""
        from src.model.architecture import create_model

        model = create_model("resnet50", pretrained=False, device=self.device)
        model.eval()
        with torch.no_grad():
            out = model(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, 10))

    def test_freeze_unfreeze_backbone(self):
        """Freezing should set all backbone params to requires_grad=False."""
        from src.model.architecture import create_model

        model = create_model("efficientnet_b0", pretrained=False, device=self.device)
        model.freeze_backbone()
        frozen_params = [p for p in model.backbone.parameters() if p.requires_grad]
        self.assertEqual(len(frozen_params), 0)

        model.unfreeze_backbone()
        unfrozen_params = [p for p in model.backbone.parameters() if p.requires_grad]
        self.assertGreater(len(unfrozen_params), 0)

    def test_predict_proba_sums_to_one(self):
        """Softmax probabilities should sum to 1 for each sample."""
        from src.model.architecture import create_model

        model = create_model("efficientnet_b0", pretrained=False, device=self.device)
        model.eval()
        probs = model.predict_proba(self.dummy_input)
        row_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(self.batch_size), atol=1e-5))

    def test_checkpoint_save_load(self):
        """Saved checkpoint should restore model weights correctly."""
        import tempfile
        import os
        from src.model.architecture import create_model, save_checkpoint, load_checkpoint

        model = create_model("efficientnet_b0", pretrained=False, device=self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pth"
            save_checkpoint(model, optimizer, epoch=5, metrics={"val_acc": 0.9}, save_path=str(ckpt_path))
            self.assertTrue(ckpt_path.exists())

            model2 = create_model("efficientnet_b0", pretrained=False, device=self.device)
            model2 = load_checkpoint(model2, str(ckpt_path), self.device)

            # Verify weights match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


# ─────────────────────────────────────────────
# Loss Function & Metrics Tests
# ─────────────────────────────────────────────

class TestLossAndMetrics(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.logits = torch.randn(16, 10)
        self.targets = torch.randint(0, 10, (16,))

    def test_label_smoothing_loss(self):
        """Label smoothing loss should be positive scalar."""
        from src.model.architecture import LabelSmoothingCrossEntropy

        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(self.logits, self.targets)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)

    def test_metrics_update_compute(self):
        """Metrics should update and compute without error."""
        from src.model.architecture import ModelMetrics

        metrics = ModelMetrics(num_classes=10, device=self.device)
        metrics.update(self.logits, self.targets)
        results = metrics.compute()

        self.assertIn("accuracy_top1", results)
        self.assertIn("f1_macro", results)
        self.assertIn("auroc", results)
        self.assertIn("confusion_matrix", results)

        acc = results["accuracy_top1"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_metrics_reset(self):
        """Reset should clear accumulated state."""
        from src.model.architecture import ModelMetrics

        metrics = ModelMetrics(num_classes=10, device=self.device)
        metrics.update(self.logits, self.targets)
        metrics.reset()
        # After reset, recompute should work with new data
        metrics.update(self.logits, self.targets)
        results = metrics.compute()
        self.assertIn("accuracy_top1", results)


# ─────────────────────────────────────────────
# Grad-CAM Tests
# ─────────────────────────────────────────────

class TestGradCAM(unittest.TestCase):

    def setUp(self):
        from src.model.architecture import create_model

        self.device = torch.device("cpu")
        self.model = create_model("efficientnet_b0", pretrained=False, device=self.device)

    def test_gradcam_generates(self):
        """Grad-CAM should generate a 2D heatmap array."""
        from src.explainability.gradcam import GradCAM

        gradcam = GradCAM(self.model)
        input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        cam, target_class, confidence = gradcam.generate(input_tensor)

        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(cam.ndim, 2)
        self.assertIsInstance(target_class, int)
        self.assertIn(target_class, range(10))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_overlay_produces_rgb(self):
        """Overlay should produce uint8 RGB image."""
        from src.explainability.gradcam import GradCAM, overlay_gradcam

        gradcam = GradCAM(self.model)
        input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        cam, _, _ = gradcam.generate(input_tensor)

        dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        overlay = overlay_gradcam(dummy_img, cam)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_explainable_predictor(self):
        """ExplainablePredictor should return full result dict."""
        from src.explainability.gradcam import ExplainablePredictor

        predictor = ExplainablePredictor(self.model, device=self.device)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        result = predictor.predict(dummy_img, generate_cam=True)

        self.assertIn("predicted_class", result)
        self.assertIn("predicted_label", result)
        self.assertIn("confidence", result)
        self.assertIn("is_distracted", result)
        self.assertIn("top_k_predictions", result)
        self.assertIn("cam", result)
        self.assertIn("cam_overlay", result)
        self.assertIn(result["predicted_class"], range(10))
        self.assertEqual(len(result["top_k_predictions"]), 3)


# ─────────────────────────────────────────────
# Training Tests
# ─────────────────────────────────────────────

class TestTrainer(unittest.TestCase):

    def test_early_stopping(self):
        """Early stopping should trigger after patience epochs with no improvement."""
        from src.training.trainer import EarlyStopping

        stopper = EarlyStopping(patience=3)
        scores = [0.5, 0.55, 0.54, 0.53, 0.52]  # No improvement after 0.55
        results = [stopper(s) for s in scores]
        self.assertIn(True, results)  # Should stop at some point

    def test_early_stopping_improves(self):
        """Early stopping should not trigger when accuracy keeps improving."""
        from src.training.trainer import EarlyStopping

        stopper = EarlyStopping(patience=3)
        for acc in [0.5, 0.6, 0.7, 0.8, 0.9]:
            stopped = stopper(acc)
            self.assertFalse(stopped)

    def test_training_config_from_dict(self):
        """TrainingConfig should accept override dict."""
        from src.training.trainer import TrainingConfig

        config = TrainingConfig({"epochs": 50, "learning_rate": 0.01})
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.learning_rate, 0.01)

    def test_minimal_training_run(self):
        """
        Run one epoch of training on synthetic data.
        Verifies the training loop executes without errors.
        """
        import tempfile
        from src.data.dataset import generate_synthetic_dataset, create_dataloaders
        from src.training.trainer import TrainingConfig, Trainer

        tmp_dir = tempfile.mkdtemp()
        data_dir = generate_synthetic_dataset(tmp_dir, samples_per_class=12)

        config = TrainingConfig({
            "architecture": "efficientnet_b0",
            "epochs": 2,
            "batch_size": 4,
            "num_workers": 0,
            "pretrained": False,
            "mixed_precision": False,
            "freeze_backbone_epochs": 0,
            "early_stopping_patience": 10,
            "output_dir": str(Path(tmp_dir) / "models"),
            "run_name": "unit_test",
            "experiment_name": "unit_tests",
        })

        dataloaders = create_dataloaders(data_dir, batch_size=4, num_workers=0)
        trainer = Trainer(config)
        best_metrics = trainer.train(dataloaders)

        self.assertIn("accuracy_top1", best_metrics)
        self.assertGreaterEqual(best_metrics["accuracy_top1"], 0)


# ─────────────────────────────────────────────
# Visualization Tests
# ─────────────────────────────────────────────

class TestVisualizations(unittest.TestCase):

    def test_training_curves_generates(self):
        """Training curves plot should save to disk."""
        import tempfile
        from src.training.visualizations import plot_training_curves

        history = [
            {"epoch": e+1, "train_loss": 2.0 - 0.05*e, "val_loss": 2.1 - 0.04*e,
             "train_acc": 0.3 + 0.03*e, "val_acc": 0.28 + 0.025*e,
             "val_f1": 0.27 + 0.024*e, "lr": 1e-3 * (0.9**e)}
            for e in range(15)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            with open(history_path, "w") as f:
                json.dump(history, f)

            fig = plot_training_curves(str(history_path), save_dir=tmpdir)
            self.assertIsNotNone(fig)
            import matplotlib.pyplot as plt
            plt.close("all")


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
