# 🚗 DriveSafe-AI
**Distracted Driver Detection using EfficientNet-B4 + MobileViT-S | CMPE 258 · SJSU · Spring 2025**

> **Authors:** Keerthana Murulidharan · Yashaswini Dinesh

---

## 🌐 Live Web App
**[drivesafe-ai.netlify.app](https://drivesafe-ai.netlify.app)** — upload a dashcam image, get instant prediction + Grad-CAM heatmap

## 📽️ Demo Video
**[▶ Watch on YouTube](#)** ← *(paste link after recording)*

## 📓 Colab Notebook
**[Open in Google Colab](https://colab.research.google.com/github/YOUR_USERNAME/drivesafe-ai/blob/main/DriveSafe_AI_CMPE258.ipynb)**

---

## 🏆 Results

| Model | Val Accuracy | F1 | AUC-ROC | Params |
|-------|:-:|:-:|:-:|:-:|
| **EfficientNet-B4** | **96.43%** | **0.9641** | **0.9912** | 19.3M |
| MobileViT-S | 91.98% | 0.9196 | 0.9724 | 5.7M |

---

## 📦 Datasets

| Dataset | Link | Size |
|---------|------|------|
| State Farm Distracted Driver | [Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection) | 22,424 images |
| AUC Distracted Driver | [AUC](https://heshameraqi.github.io/distraction_detection) | ~14,000 images |

---

## 🚀 How to Run

### Option 1 — Web App (no setup needed)
Go to **[drivesafe-ai.netlify.app](https://drivesafe-ai.netlify.app)** and upload any dashcam image.

### Option 2 — Colab Notebook
1. Click **Open in Colab** above
2. Runtime → Change runtime type → **T4 GPU**
3. Mount Google Drive with dataset:
```python
MOUNT_DRIVE = True
ZIP_PATH = "/content/drive/MyDrive/imgs.zip"
CSV_PATH = "/content/drive/MyDrive/driver_imgs_list.csv"
```
4. Run All Cells (`Runtime → Run all`)

### Option 3 — Local
```bash
git clone https://github.com/YOUR_USERNAME/drivesafe-ai.git
cd drivesafe-ai
pip install -r requirements.txt
python src/train.py --sf_root data/state_farm --auc_root data/auc_dataset
```

---

## 📁 Repo Structure
```
drivesafe_ai/
├── index.html                  ← Web app (deploy to Netlify)
├── DriveSafe_AI_CMPE258.ipynb  ← Full Colab notebook
├── src/
│   ├── train.py                ← Training pipeline
│   └── gradcam_inference.py    ← Grad-CAM visualisation
├── results/                    ← All 10 graphs
├── requirements.txt
└── README.md
```

---

## 🔑 Key Techniques
- **Dual-dataset fusion** — State Farm + AUC (28K+ images)
- **Driver-aware splitting** — no leakage between train/val
- **Progressive fine-tuning** — 3-stage backbone unfreezing
- **WeightedRandomSampler** — handles class imbalance
- **Label Smoothing + Focal Loss** — calibration & hard example focus
- **Grad-CAM** — visual explainability on last conv block
- **Albumentations** — 8-transform augmentation pipeline
- **Mixed precision (AMP)** — 2× faster training

---

## 📋 Requirements
```
torch>=2.0.0  torchvision>=0.15.0  timm>=0.9.0
albumentations>=2.0.0  scikit-learn>=1.2.0
matplotlib>=3.7.0  seaborn>=0.12.0  pandas numpy pillow tqdm
```

---

*CMPE 258 Deep Learning · San Jose State University · Spring 2025*
