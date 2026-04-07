# Base image: PyTorch with CUDA 12.1
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    git wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' point to python3
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY Q1/ Q1/
COPY Q2/ Q2/
COPY push_to_hub.py .
COPY README.md .

# Create results directories
RUN mkdir -p /workspace/results/Q1 /workspace/results/Q2i /workspace/results/Q2ii

CMD ["python3", "Q1/train_vit_lora.py", "--help"]
