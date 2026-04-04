# Base image: PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Reproduce the repo structure inside the image ──────────────────────────
# Copy each sub-folder explicitly so paths match README run commands:
#   python Q1/train_vit_lora.py ...
#   python Q2/fgsm_attack.py   ...
#   python Q2/adversarial_detection.py ...
#   python push_to_hub.py ...
COPY Q1/ Q1/
COPY Q2/ Q2/
COPY push_to_hub.py .
COPY README.md .

# Results directory (overridden by -v mount at runtime)
RUN mkdir -p /workspace/results/Q1 /workspace/results/Q2i /workspace/results/Q2ii

# WandB key can be injected at runtime:
#   docker run -e WANDB_API_KEY=<your_key> ...
# OR disable WandB entirely with --no_wandb flag (default CMD uses it)
ENV WANDB_API_KEY=""

# Default: show Q1 help  (override at `docker run` time)
CMD ["python", "Q1/train_vit_lora.py", "--help"]