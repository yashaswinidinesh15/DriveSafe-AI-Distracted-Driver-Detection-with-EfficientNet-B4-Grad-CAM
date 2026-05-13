FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install PyTorch CPU-only (smaller image for inference)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    timm \
    torchmetrics \
    mlflow \
    numpy \
    pandas \
    scikit-learn \
    Pillow \
    matplotlib \
    seaborn \
    flask \
    flask-cors \
    gradio \
    opencv-python-headless \
    pyyaml \
    tqdm

# Copy application code
COPY src/ ./src/
COPY webapp/ ./webapp/
COPY configs/ ./configs/
COPY mlops/ ./mlops/
COPY scripts/ ./scripts/

# Create model directory
RUN mkdir -p models logs data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/best_model.pth
ENV MODEL_ARCH=efficientnet_b3
ENV PORT=7860

EXPOSE 7860

# Launch Gradio app
CMD ["python", "webapp/gradio_app.py"]
