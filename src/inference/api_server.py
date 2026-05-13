"""
Inference REST API Server
=========================
Production Flask API for distracted driver detection inference.
Endpoints:
  POST /predict        - Single image prediction with Grad-CAM
  POST /predict/batch  - Batch prediction (no Grad-CAM)
  GET  /health         - Health check
  GET  /model/info     - Model metadata
  GET  /classes        - Class definitions
"""

import io
import os
import sys
import base64
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────

def create_app(model_path: str = None, architecture: str = "efficientnet_b3") -> Flask:
    """Create and configure Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.model.architecture import create_model, load_checkpoint
        from src.explainability.gradcam import ExplainablePredictor

        model = create_model(architecture, pretrained=False, device=device)

        if model_path and Path(model_path).exists():
            model = load_checkpoint(model, model_path, device)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning("No model checkpoint found. Using untrained weights.")

        app.predictor = ExplainablePredictor(model, device=device)
        app.model_path = model_path or "untrained"
        app.architecture = architecture
        app.model_loaded = True
        logger.info(f"Model ready on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.model_loaded = False

    return app


app = create_app(
    model_path=os.environ.get("MODEL_PATH", "models/best_model.pth"),
    architecture=os.environ.get("MODEL_ARCH", "efficientnet_b3"),
)

CLASS_DEFINITIONS = {
    0: {"code": "c0", "name": "Safe Driving", "risk": "none", "color": "#27ae60"},
    1: {"code": "c1", "name": "Texting (Right Hand)", "risk": "high", "color": "#e74c3c"},
    2: {"code": "c2", "name": "Phone Call (Right Hand)", "risk": "high", "color": "#e74c3c"},
    3: {"code": "c3", "name": "Texting (Left Hand)", "risk": "high", "color": "#e74c3c"},
    4: {"code": "c4", "name": "Phone Call (Left Hand)", "risk": "high", "color": "#e74c3c"},
    5: {"code": "c5", "name": "Radio Adjusting", "risk": "medium", "color": "#f39c12"},
    6: {"code": "c6", "name": "Drinking", "risk": "medium", "color": "#f39c12"},
    7: {"code": "c7", "name": "Reaching Behind", "risk": "high", "color": "#e74c3c"},
    8: {"code": "c8", "name": "Hair / Makeup", "risk": "medium", "color": "#f39c12"},
    9: {"code": "c9", "name": "Talking to Passenger", "risk": "low", "color": "#3498db"},
}


def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy image to base64 encoded PNG string."""
    pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_image_from_request() -> Image.Image:
    """Parse image from request (file upload or base64)."""
    if "file" in request.files:
        file = request.files["file"]
        return Image.open(file.stream).convert("RGB")

    if request.is_json:
        data = request.get_json()
        if "image_base64" in data:
            img_bytes = base64.b64decode(data["image_base64"])
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError("No image found in request. Send 'file' or 'image_base64'.")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if app.model_loaded else "degraded",
        "model_loaded": app.model_loaded,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "timestamp": time.time(),
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({"classes": CLASS_DEFINITIONS})


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "architecture": getattr(app, "architecture", "unknown"),
        "num_classes": 10,
        "model_path": getattr(app, "model_path", "unknown"),
        "class_definitions": CLASS_DEFINITIONS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single image prediction with Grad-CAM.

    Request: multipart/form-data with 'file' field
             OR application/json with 'image_base64' field

    Response: {
        predicted_class, predicted_label, confidence,
        is_distracted, risk_level,
        top_k_predictions: [...],
        all_probabilities: [...],
        cam_overlay_base64: "...",
        class_definition: {...}
    }
    """
    if not app.model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    start_time = time.time()

    try:
        pil_image = parse_image_from_request()
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = app.predictor.predict(pil_image, top_k=3, generate_cam=True)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    inference_time = time.time() - start_time
    predicted_class = result["predicted_class"]
    class_def = CLASS_DEFINITIONS.get(predicted_class, {})

    response = {
        "predicted_class": predicted_class,
        "predicted_label": result["predicted_label"],
        "confidence": round(result["confidence"], 4),
        "is_distracted": result["is_distracted"],
        "risk_level": class_def.get("risk", "unknown"),
        "risk_color": class_def.get("color", "#777"),
        "top_k_predictions": result["top_k_predictions"],
        "all_probabilities": result["all_probabilities"],
        "class_definition": class_def,
        "inference_time_ms": round(inference_time * 1000, 2),
    }

    # Add Grad-CAM if available
    if "cam_overlay" in result and result["cam_overlay"] is not None:
        response["cam_overlay_base64"] = image_to_base64(result["cam_overlay"])

    return jsonify(response)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction from multiple uploaded files (no Grad-CAM)."""
    if not app.model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    images = []
    filenames = []
    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
            images.append(img)
            filenames.append(f.filename)
        except Exception as e:
            logger.warning(f"Failed to load {f.filename}: {e}")

    if not images:
        return jsonify({"error": "No valid images"}), 400

    results = app.predictor.predict_batch(images)

    response = []
    for fname, result in zip(filenames, results):
        pred_class = result["predicted_class"]
        class_def = CLASS_DEFINITIONS.get(pred_class, {})
        response.append({
            "filename": fname,
            "predicted_class": pred_class,
            "predicted_label": result["predicted_label"],
            "confidence": round(result["confidence"], 4),
            "is_distracted": result["is_distracted"],
            "risk_level": class_def.get("risk", "unknown"),
        })

    distracted_count = sum(1 for r in response if r["is_distracted"])
    return jsonify({
        "predictions": response,
        "total": len(response),
        "distracted_count": distracted_count,
        "distracted_percentage": round(distracted_count / len(response) * 100, 1),
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
