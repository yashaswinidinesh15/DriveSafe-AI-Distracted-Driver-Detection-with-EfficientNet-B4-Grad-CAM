"""
MLOps Pipeline Orchestrator
============================
End-to-end ML pipeline with MLflow Projects structure.
Implements MLOps Maturity Level 2:
  - Automated training with experiment tracking
  - Centralized model registry
  - Reproducible runs with versioned artifacts
  - Model evaluation and promotion gates

Pipeline stages:
  1. data_ingestion  → download and validate dataset
  2. data_processing → split, augment, compute statistics
  3. model_training  → train with full metric logging
  4. model_evaluation → evaluate on test set, generate reports
  5. model_registry  → register best model in MLflow registry
  6. ablation_study  → systematic hyperparameter exploration
"""

import os
import sys
import json
import logging
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────

class DataIngestionStep:
    """
    Step 1: Data ingestion and validation.
    Downloads State Farm dataset from Kaggle or uses local path.
    Validates data integrity and class distribution.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.raw_dir = Path(config.get("raw_dir", "data/raw"))
        self.processed_dir = Path(config.get("processed_dir", "data/processed"))

    def run(self, use_synthetic: bool = False) -> Dict:
        """
        Execute data ingestion.

        If Kaggle credentials available: download from Kaggle
        Else: generate synthetic dataset for pipeline testing
        """
        logger.info("=== STEP 1: Data Ingestion ===")

        sys.path.insert(0, str(ROOT))
        from src.data.dataset import (
            generate_synthetic_dataset,
            compute_dataset_statistics,
            split_dataset,
        )

        if use_synthetic:
            logger.info("Using synthetic dataset (Kaggle not configured)")
            samples = self.config.get("synthetic_samples_per_class", 50)
            # generate_synthetic_dataset creates train/val/test inside the given dir
            generate_synthetic_dataset(str(self.raw_dir), samples)
            stats = compute_dataset_statistics(str(self.raw_dir))
        else:
            # Attempt Kaggle download
            try:
                self._download_from_kaggle()
                stats = compute_dataset_statistics(str(self.raw_dir))
            except Exception as e:
                logger.warning(f"Kaggle download failed: {e}. Falling back to synthetic.")
                return self.run(use_synthetic=True)

        output = {
            "status": "success",
            "data_dir": str(self.raw_dir),
            "statistics": stats,
        }

        # Log to MLflow
        with mlflow.start_run(run_name="data_ingestion", nested=True):
            for split, info in stats.items():
                mlflow.log_metric(f"{split}_samples", info.get("total", 0))
            mlflow.log_param("data_source", "synthetic" if use_synthetic else "kaggle")

        logger.info(f"Data ingestion complete: {stats}")
        return output

    def _download_from_kaggle(self):
        """Download State Farm dataset using Kaggle API."""
        logger.info("Downloading from Kaggle: state-farm-distracted-driver-detection")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "state-farm-distracted-driver-detection",
                "-p", str(self.raw_dir),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle download failed: {result.stderr}")
        logger.info("Kaggle download complete. Extracting...")
        # Unzip downloaded file
        for zipfile in self.raw_dir.glob("*.zip"):
            shutil.unpack_archive(str(zipfile), str(self.raw_dir))


class DataProcessingStep:
    """
    Step 2: Data processing and splitting.
    Applies stratified splits and saves manifests.
    """

    def __init__(self, config: Dict):
        self.config = config

    def run(self, data_dir: str) -> Dict:
        logger.info("=== STEP 2: Data Processing ===")

        sys.path.insert(0, str(ROOT))
        from src.data.dataset import split_dataset, compute_dataset_statistics, generate_synthetic_dataset

        processed_dir = Path(self.config.get("processed_dir", "data/processed"))

        # Check if data_dir already has train/val/test structure
        data_path = Path(data_dir)
        has_splits = all((data_path / s).exists() for s in ["train", "val", "test"])

        if has_splits:
            logger.info("Pre-split data detected. Skipping split step.")
            processed_dir = Path(data_dir)  # already split, use as-is
            stats = compute_dataset_statistics(data_dir)
        else:
            # Try to find raw images
            raw_train = data_path / "imgs" / "train"
            if raw_train.exists():
                source = str(raw_train)
            else:
                # Generate synthetic for CI
                logger.info("Generating synthetic dataset for pipeline test...")
                source_dir = generate_synthetic_dataset("data/synthetic_pipeline", 30)
                # Use the already-split synthetic data
                processed_dir.mkdir(parents=True, exist_ok=True)
                for split in ["train", "val", "test"]:
                    src = Path(source_dir) / split
                    dst = processed_dir / split
                    if src.exists():
                        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
                stats = compute_dataset_statistics(str(processed_dir))

                with mlflow.start_run(run_name="data_processing", nested=True):
                    for split, info in stats.items():
                        mlflow.log_metric(f"processed_{split}_samples", info.get("total", 0))
                    mlflow.log_param("train_ratio", 0.70)
                    mlflow.log_param("val_ratio", 0.15)
                    mlflow.log_param("test_ratio", 0.15)
                    mlflow.log_param("stratified", True)

                return {"status": "success", "processed_dir": str(processed_dir), "statistics": stats}

            counts = split_dataset(
                source_dir=source,
                output_dir=str(processed_dir),
                train_ratio=self.config.get("train_split", 0.70),
                val_ratio=self.config.get("val_split", 0.15),
                test_ratio=self.config.get("test_split", 0.15),
            )
            stats = compute_dataset_statistics(str(processed_dir))

        with mlflow.start_run(run_name="data_processing", nested=True):
            for split, info in stats.items():
                mlflow.log_metric(f"processed_{split}_samples", info.get("total", 0))

        return {"status": "success", "processed_dir": str(processed_dir), "statistics": stats}


class ModelTrainingStep:
    """
    Step 3: Model training with full MLflow tracking.
    """

    def __init__(self, config: Dict):
        self.config = config

    def run(self, data_dir: str) -> Dict:
        logger.info("=== STEP 3: Model Training ===")

        sys.path.insert(0, str(ROOT))
        from src.training.trainer import TrainingConfig, Trainer
        from src.data.dataset import create_dataloaders

        training_config = TrainingConfig(self.config)
        training_config.data_dir = data_dir
        training_config.run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=training_config.batch_size,
            image_size=training_config.image_size,
            num_workers=training_config.num_workers,
            use_weighted_sampler=training_config.use_weighted_sampler,
        )

        trainer = Trainer(training_config)
        best_metrics = trainer.train(dataloaders)

        return {
            "status": "success",
            "best_metrics": best_metrics,
            "model_path": str(Path(training_config.output_dir) / "best_model.pth"),
            "run_name": training_config.run_name,
        }


class ModelEvaluationStep:
    """
    Step 4: Comprehensive test set evaluation.
    Generates confusion matrix, per-class metrics, and Grad-CAM samples.
    """

    def __init__(self, config: Dict):
        self.config = config

    def run(self, model_path: str, data_dir: str) -> Dict:
        logger.info("=== STEP 4: Model Evaluation ===")

        import torch
        sys.path.insert(0, str(ROOT))
        from src.model.architecture import create_model, load_checkpoint, ModelMetrics
        from src.data.dataset import IDX_TO_NAME
        from src.data.dataset import create_dataloaders
        from src.training.trainer import evaluate, TrainingConfig

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch = self.config.get("architecture", "efficientnet_b3")
        model = create_model(arch, pretrained=False, device=device)

        if Path(model_path).exists():
            model = load_checkpoint(model, model_path, device)

        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=self.config.get("batch_size", 32),
            image_size=self.config.get("image_size", 224),
            num_workers=0,
        )

        import torch.nn as nn
        loss_fn = nn.CrossEntropyLoss()
        cfg = TrainingConfig(self.config)

        test_metrics = evaluate(model, dataloaders["test"], loss_fn, cfg, device, "test")

        with mlflow.start_run(run_name="model_evaluation", nested=True):
            mlflow.log_metrics({
                "test_accuracy_top1": test_metrics["accuracy_top1"],
                "test_accuracy_top3": test_metrics.get("accuracy_top3", 0),
                "test_f1_macro": test_metrics["f1_macro"],
                "test_precision": test_metrics["precision_macro"],
                "test_recall": test_metrics["recall_macro"],
                "test_auroc": test_metrics["auroc"],
                "test_loss": test_metrics["loss"],
            })

            # Log per-class accuracy
            per_class_acc = test_metrics.get("per_class_accuracy", [])
            for i, acc in enumerate(per_class_acc):
                safe_name = IDX_TO_NAME.get(i, str(i)).replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                mlflow.log_metric(f"test_acc_class_{safe_name}", acc)

        logger.info(f"Test Results:")
        logger.info(f"  Top-1 Accuracy: {test_metrics['accuracy_top1']:.4f}")
        logger.info(f"  F1 Macro:       {test_metrics['f1_macro']:.4f}")
        logger.info(f"  AUROC:          {test_metrics['auroc']:.4f}")

        return {"status": "success", "test_metrics": test_metrics}


class ModelRegistryStep:
    """
    Step 5: Register model in MLflow Model Registry.
    Promotes model if it meets quality gates.
    """

    QUALITY_GATES = {
        "min_accuracy": 0.70,
        "min_f1": 0.65,
    }

    def run(self, model_path: str, test_metrics: Dict) -> Dict:
        logger.info("=== STEP 5: Model Registry ===")

        client = MlflowClient()

        # Quality gates
        acc = test_metrics.get("accuracy_top1", 0)
        f1 = test_metrics.get("f1_macro", 0)
        gates_passed = (
            acc >= self.QUALITY_GATES["min_accuracy"] and
            f1 >= self.QUALITY_GATES["min_f1"]
        )

        if not gates_passed:
            logger.warning(
                f"Quality gates FAILED: acc={acc:.4f} (min={self.QUALITY_GATES['min_accuracy']}), "
                f"f1={f1:.4f} (min={self.QUALITY_GATES['min_f1']})"
            )
            return {"status": "gates_failed", "metrics": test_metrics}

        # Register model
        model_name = "distracted-driver-detector"
        try:
            client.create_registered_model(
                model_name,
                description="EfficientNet-B3 based distracted driver behavior classifier",
            )
        except Exception:
            pass  # Already exists

        # Log and register
        with mlflow.start_run(run_name="model_registration", nested=True):
            import torch
            sys.path.insert(0, str(ROOT))
            from src.model.architecture import create_model, load_checkpoint

            device = torch.device("cpu")
            model = create_model("efficientnet_b3", pretrained=False, device=device)
            if Path(model_path).exists():
                model = load_checkpoint(model, model_path, device)

            mlflow.log_metrics(test_metrics)
            result = mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=model_name,
            )

        logger.info(f"Model registered: {model_name}")
        return {
            "status": "success",
            "model_name": model_name,
            "gates_passed": True,
            "test_accuracy": acc,
            "test_f1": f1,
        }


# ─────────────────────────────────────────────
# Full Pipeline Runner
# ─────────────────────────────────────────────

class MLPipeline:
    """
    Orchestrates the full end-to-end ML pipeline.
    Each step logs to MLflow under a parent run.
    """

    def __init__(self, config: Dict):
        self.config = config
        mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(config.get("experiment_name", "distracted-driver-detection"))

    def run(
        self,
        use_synthetic: bool = True,
        skip_training: bool = False,
        skip_ablation: bool = True,
    ) -> Dict:
        """Execute all pipeline stages."""
        pipeline_start = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info("DISTRACTED DRIVER DETECTION — ML PIPELINE")
        logger.info(f"Started: {pipeline_start}")
        logger.info(f"{'='*60}\n")

        results = {}

        with mlflow.start_run(run_name=f"pipeline_{pipeline_start.strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params(self.config)

            # Step 1: Data Ingestion
            ingestion = DataIngestionStep(self.config)
            results["ingestion"] = ingestion.run(use_synthetic=use_synthetic)

            # Step 2: Data Processing
            processing = DataProcessingStep(self.config)
            results["processing"] = processing.run(results["ingestion"]["data_dir"])

            processed_dir = results["processing"]["processed_dir"]

            # Step 3: Training
            if not skip_training:
                training = ModelTrainingStep(self.config)
                results["training"] = training.run(processed_dir)
                model_path = results["training"]["model_path"]
            else:
                model_path = self.config.get("model_path", "models/best_model.pth")
                results["training"] = {"status": "skipped", "model_path": model_path}

            # Step 4: Evaluation
            evaluation = ModelEvaluationStep(self.config)
            results["evaluation"] = evaluation.run(model_path, processed_dir)

            # Step 5: Registry
            registry = ModelRegistryStep()
            results["registry"] = registry.run(
                model_path,
                results["evaluation"].get("test_metrics", {}),
            )

            # Step 6: Ablation Study (optional)
            if not skip_ablation:
                results["ablation"] = self._run_ablation(processed_dir)

            # Save pipeline results
            results_path = Path("mlops/pipeline_results.json")
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump({k: v for k, v in results.items() if k != "ablation"}, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))

            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            mlflow.log_metric("pipeline_duration_seconds", pipeline_duration)
            logger.info(f"\nPipeline complete in {pipeline_duration:.1f}s")

        return results

    def _run_ablation(self, data_dir: str) -> Dict:
        """Run ablation studies."""
        logger.info("=== STEP 6: Ablation Study ===")
        sys.path.insert(0, str(ROOT))
        from src.training.trainer import AblationStudy, TrainingConfig
        from src.data.dataset import create_dataloaders

        base_config = TrainingConfig(self.config)
        base_config.data_dir = data_dir
        base_config.epochs = 5  # Short runs for ablation

        dataloaders = create_dataloaders(
            data_dir=data_dir,
            batch_size=base_config.batch_size,
            image_size=base_config.image_size,
            num_workers=0,
        )

        ablation = AblationStudy(base_config, dataloaders)
        arch_results = ablation.run_architecture_ablation()
        ablation.save_results("mlops/ablation_results.json")

        return {"status": "success", "results": arch_results}


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distracted Driver Detection MLOps Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-ablation", action="store_true", default=True)
    parser.add_argument("--run-ablation", action="store_true", help="Run ablation study")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
    )

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    flat_config = {}
    for section in config.values():
        if isinstance(section, dict):
            flat_config.update(section)

    pipeline = MLPipeline(flat_config)
    results = pipeline.run(
        use_synthetic=args.synthetic,
        skip_training=args.skip_training,
        skip_ablation=not args.run_ablation,
    )

    print("\n=== PIPELINE SUMMARY ===")
    for stage, result in results.items():
        status = result.get("status", "unknown") if isinstance(result, dict) else "done"
        print(f"  {stage:20s}: {status}")


if __name__ == "__main__":
    main()
