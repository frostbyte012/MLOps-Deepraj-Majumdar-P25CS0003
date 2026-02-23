# ── Dockerfile (Training) ─────────────────────────────────────────────────────
# Base image: Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb .

# Create results directory
RUN mkdir -p results logs saved_model

# Default command: run training
CMD ["python", "src/train.py", "--epochs", "3", "--batch_size", "10"]