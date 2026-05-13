"""
Deploy to HuggingFace Spaces
Pushes the Gradio webapp to HF Spaces for public demo.
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
SPACE_NAME = "distracted-driver-detection"
REPO_ID = f"{HF_USERNAME}/{SPACE_NAME}"

ROOT = Path(__file__).parent.parent


def deploy():
    if not HF_TOKEN:
        print("HF_TOKEN not set. Skipping deployment.")
        return

    api = HfApi(token=HF_TOKEN)

    # Create space if it doesn't exist
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            token=HF_TOKEN,
        )
        print(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"Space creation: {e}")

    # Files to upload
    files_to_upload = [
        ("webapp/gradio_app.py", "app.py"),
        ("src/model/architecture.py", "src/model/architecture.py"),
        ("src/explainability/gradcam.py", "src/explainability/gradcam.py"),
        ("src/data/dataset.py", "src/data/dataset.py"),
        ("configs/config.yaml", "configs/config.yaml"),
        ("requirements.txt", "requirements.txt"),
    ]

    # Create HF-specific requirements
    hf_requirements = """torch
torchvision
timm
torchmetrics
numpy
pandas
scikit-learn
Pillow
matplotlib
seaborn
gradio>=4.0.0
opencv-python-headless
pyyaml
tqdm
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(hf_requirements)
        hf_req_path = f.name

    # Upload files
    for local_path, remote_path in files_to_upload:
        local_full = ROOT / local_path
        if local_full.exists():
            api.upload_file(
                path_or_fileobj=str(local_full),
                path_in_repo=remote_path,
                repo_id=REPO_ID,
                repo_type="space",
                token=HF_TOKEN,
            )
            print(f"Uploaded: {remote_path}")

    # Upload special HF requirements
    api.upload_file(
        path_or_fileobj=hf_req_path,
        path_in_repo="requirements.txt",
        repo_id=REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
    )

    # Upload model if available
    model_path = ROOT / "models" / "best_model.pth"
    if model_path.exists():
        print("Uploading model checkpoint...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="models/best_model.pth",
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
        )

    print(f"\n✅ Deployed to: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    deploy()
