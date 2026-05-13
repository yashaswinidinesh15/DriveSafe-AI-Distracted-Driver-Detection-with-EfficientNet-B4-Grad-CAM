"""
Gradio Web Demo — Distracted Driver Detection
==============================================
Production-grade inference demo with:
- Image upload and webcam capture
- Real-time prediction with confidence scores
- Grad-CAM visualization
- Risk level assessment
- Batch analysis mode
"""

import sys
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model.architecture import create_model, load_checkpoint, IDX_TO_NAME
from src.explainability.gradcam import ExplainablePredictor, create_gradcam_figure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DEFINITIONS = {
    0: {"name": "Safe Driving", "risk": "✅ No Risk", "color": "#27ae60", "emoji": "✅"},
    1: {"name": "Texting (Right Hand)", "risk": "🚨 High Risk", "color": "#e74c3c", "emoji": "📱"},
    2: {"name": "Phone Call (Right Hand)", "risk": "🚨 High Risk", "color": "#e74c3c", "emoji": "📞"},
    3: {"name": "Texting (Left Hand)", "risk": "🚨 High Risk", "color": "#e74c3c", "emoji": "📱"},
    4: {"name": "Phone Call (Left Hand)", "risk": "🚨 High Risk", "color": "#e74c3c", "emoji": "📞"},
    5: {"name": "Radio Adjusting", "risk": "⚠️ Medium Risk", "color": "#f39c12", "emoji": "📻"},
    6: {"name": "Drinking", "risk": "⚠️ Medium Risk", "color": "#f39c12", "emoji": "🥤"},
    7: {"name": "Reaching Behind", "risk": "🚨 High Risk", "color": "#e74c3c", "emoji": "🙆"},
    8: {"name": "Hair / Makeup", "risk": "⚠️ Medium Risk", "color": "#f39c12", "emoji": "💄"},
    9: {"name": "Talking to Passenger", "risk": "ℹ️ Low Risk", "color": "#3498db", "emoji": "💬"},
}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_predictor(
    model_path: Optional[str] = None,
    architecture: str = "efficientnet_b3",
) -> ExplainablePredictor:
    """Load model and create predictor."""
    model = create_model(architecture, pretrained=False, device=DEVICE)

    if model_path and Path(model_path).exists():
        model = load_checkpoint(model, model_path, DEVICE)
        logger.info(f"Loaded checkpoint: {model_path}")
    else:
        logger.warning("No checkpoint found. Using random weights (demo mode).")

    return ExplainablePredictor(model, device=DEVICE)


MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "best_model.pth"))
PREDICTOR = load_predictor(MODEL_PATH)


# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────

def predict_image(image: np.ndarray) -> Tuple:
    """
    Main prediction function for Gradio interface.

    Args:
        image: numpy array from Gradio image input

    Returns:
        Tuple of (cam_overlay_image, result_text, probabilities_plot)
    """
    if image is None:
        return None, "❌ Please upload an image.", None

    try:
        pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    except Exception as e:
        return None, f"❌ Image processing error: {str(e)}", None

    # Run prediction
    result = PREDICTOR.predict(pil_image, top_k=5, generate_cam=True)

    pred_class = result["predicted_class"]
    pred_label = result["predicted_label"]
    confidence = result["confidence"]
    class_info = CLASS_DEFINITIONS[pred_class]

    # Format result text
    risk_text = class_info["risk"]
    emoji = class_info["emoji"]

    result_text = f"""
## {emoji} {pred_label}
**Risk Level:** {risk_text}
**Confidence:** {confidence:.1%}

### Top Predictions:
"""
    for pred in result["top_k_predictions"]:
        cls_info = CLASS_DEFINITIONS.get(pred["class_idx"], {})
        e = cls_info.get("emoji", "•")
        result_text += f"\n{e} **{pred['label']}**: {pred['confidence']:.1%}"

    if result["is_distracted"]:
        result_text += f"\n\n---\n⚠️ **ALERT: Driver is distracted! Immediate attention required.**"
    else:
        result_text += f"\n\n---\n✅ **Driver appears to be focused on the road.**"

    # Grad-CAM overlay image
    cam_overlay = result.get("cam_overlay")

    # Probability chart
    fig = create_probability_chart(result["all_probabilities"])

    return cam_overlay, result_text, fig


def create_probability_chart(probs: list) -> plt.Figure:
    """Create a styled horizontal bar chart of class probabilities."""
    class_names = [CLASS_DEFINITIONS[i]["emoji"] + " " + CLASS_DEFINITIONS[i]["name"]
                   for i in range(10)]
    colors = [CLASS_DEFINITIONS[i]["color"] for i in range(10)]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    y_pos = range(len(class_names))
    bars = ax.barh(y_pos, [p * 100 for p in probs], color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(class_names, fontsize=9, color="white")
    ax.set_xlabel("Confidence (%)", color="white", fontsize=10)
    ax.set_title("Classification Probabilities", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)

    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}",
                va="center", ha="left", color="white", fontsize=8.5,
            )

    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Example Images
# ─────────────────────────────────────────────

def generate_example_images():
    """Generate synthetic example images for the demo."""
    examples = []
    colors = [
        (50, 180, 50),    # Green - safe
        (220, 50, 50),    # Red - texting
        (220, 50, 50),    # Red - phone
        (240, 180, 20),   # Orange - drinking
        (50, 100, 220),   # Blue - passenger
    ]
    labels = ["Safe Driving", "Texting", "Phone Call", "Drinking", "Talking"]

    for i, (color, label) in enumerate(zip(colors, labels)):
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        img_array[:] = color
        noise = np.random.randint(0, 40, (224, 224, 3), dtype=np.uint8)
        img_array = np.clip(img_array.astype(int) + noise - 20, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        with tempfile.NamedTemporaryFile(suffix=f"_{label}.jpg", delete=False) as tmp:
            img.save(tmp.name)
            examples.append(tmp.name)

    return examples


# ─────────────────────────────────────────────
# Gradio Interface
# ─────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    """Build the full Gradio demo interface."""

    css = """
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; margin-bottom: 20px; }
    .result-box { border-radius: 10px; padding: 15px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="🚗 Distracted Driver Detection",
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        css=css,
    ) as demo:
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="font-size:2.2em; margin-bottom:5px;">🚗 Distracted Driver Detection</h1>
            <p style="color:#888; font-size:1.1em;">
                AI-powered dashcam analysis using EfficientNet + Grad-CAM Explainability
            </p>
            <div style="display:flex; gap:10px; justify-content:center; margin-top:10px; flex-wrap:wrap;">
                <span style="background:#e74c3c; color:white; padding:4px 12px; border-radius:20px; font-size:0.85em;">🚨 High Risk Detection</span>
                <span style="background:#f39c12; color:white; padding:4px 12px; border-radius:20px; font-size:0.85em;">⚠️ Medium Risk</span>
                <span style="background:#27ae60; color:white; padding:4px 12px; border-radius:20px; font-size:0.85em;">✅ Safe Driving</span>
                <span style="background:#3498db; color:white; padding:4px 12px; border-radius:20px; font-size:0.85em;">🔍 Grad-CAM Explained</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Single Image Analysis ──
            with gr.Tab("📸 Single Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Dashcam Image",
                            type="numpy",
                            height=300,
                        )
                        analyze_btn = gr.Button(
                            "🔍 Analyze Driver Behavior",
                            variant="primary",
                            size="lg",
                        )
                        gr.Markdown("**Example images:**")
                        example_paths = generate_example_images()
                        gr.Examples(
                            examples=example_paths,
                            inputs=image_input,
                            label="",
                        )

                    with gr.Column(scale=1):
                        cam_output = gr.Image(
                            label="Grad-CAM Explanation (What the model focuses on)",
                            height=300,
                        )
                        result_output = gr.Markdown(
                            value="*Upload an image and click Analyze*",
                        )

                with gr.Row():
                    prob_chart = gr.Plot(label="Class Probability Distribution")

                analyze_btn.click(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[cam_output, result_output, prob_chart],
                )
                image_input.change(
                    fn=predict_image,
                    inputs=[image_input],
                    outputs=[cam_output, result_output, prob_chart],
                )

            # ── Tab 2: About the Model ──
            with gr.Tab("📊 Model & Dataset Info"):
                gr.Markdown("""
                ## Model Architecture

                | Component | Details |
                |-----------|---------|
                | **Backbone** | EfficientNet-B3 (pretrained on ImageNet) |
                | **Head** | Custom 2-layer MLP with BatchNorm + SiLU |
                | **Parameters** | ~12M total, ~2M trainable (Phase 1) |
                | **Input Size** | 224×224 RGB |
                | **Output** | 10-class softmax |
                | **Explainability** | Grad-CAM on last conv layer |

                ## Dataset: State Farm Distracted Driver Detection

                | Split | Samples |
                |-------|---------|
                | Train | ~15,400 (70%) |
                | Validation | ~3,300 (15%) |
                | Test | ~3,300 (15%) |

                **10 Behavior Classes:**

                | Class | Behavior | Risk Level |
                |-------|----------|------------|
                | c0 | Safe Driving | ✅ None |
                | c1 | Texting (Right Hand) | 🚨 High |
                | c2 | Phone Call (Right Hand) | 🚨 High |
                | c3 | Texting (Left Hand) | 🚨 High |
                | c4 | Phone Call (Left Hand) | 🚨 High |
                | c5 | Radio Adjusting | ⚠️ Medium |
                | c6 | Drinking | ⚠️ Medium |
                | c7 | Reaching Behind | 🚨 High |
                | c8 | Hair / Makeup | ⚠️ Medium |
                | c9 | Talking to Passenger | ℹ️ Low |

                ## Training Strategy
                - **Phase 1** (epochs 1–3): Backbone frozen, train head only
                - **Phase 2** (epoch 4+): Full fine-tuning with 10× lower backbone LR
                - **Loss**: Label Smoothing Cross-Entropy (ε=0.1)
                - **Sampler**: WeightedRandomSampler for class imbalance
                - **Augmentation**: Random flip, rotation, color jitter, random erasing

                ## Key Design Choices
                - **EfficientNet over ResNet**: Better accuracy/compute tradeoff
                - **Label smoothing**: Prevents overconfident predictions
                - **Grad-CAM**: Verifies model looks at correct body regions (hands, face)
                - **Mixed precision (AMP)**: 2× faster training without accuracy loss
                """)

            # ── Tab 3: Risk Assessment Guide ──
            with gr.Tab("⚠️ Risk Assessment Guide"):
                gr.Markdown("""
                ## Risk Level Definitions

                ### 🚨 High Risk
                Behaviors that critically impair driving ability:
                - **Texting** (both hands): Visual + manual + cognitive distraction
                - **Phone calls**: Manual distraction + attention diversion
                - **Reaching behind**: Physical posture limits vehicle control

                ### ⚠️ Medium Risk
                Behaviors that moderately impair driving:
                - **Drinking**: Temporary single-hand control
                - **Radio adjusting**: Brief visual attention shift
                - **Hair / Makeup**: Extended manual distraction

                ### ℹ️ Low Risk
                - **Talking to passenger**: Cognitive distraction only

                ### ✅ No Risk
                - **Safe driving**: Full attention on road

                ---
                ## Applications
                - **Fleet management**: Real-time monitoring of commercial drivers
                - **Insurance scoring**: Behavioral risk assessment
                - **Driver coaching**: Personalized safety feedback
                - **Regulatory compliance**: Mandatory distraction reporting
                """)

        gr.HTML("""
        <div style="text-align:center; color:#666; margin-top:20px; font-size:0.85em;">
            Deep Learning Based Distracted Driver Detection |
            EfficientNet-B3 + Grad-CAM | State Farm Dataset |
            MLOps Pipeline with MLflow + TensorBoard
        </div>
        """)

    return demo


# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
    )
