# ─────────────────────────────────────────────────────────────
# NexusVoice — Multi-stage Dockerfile
# Stage A (builder): installs all deps including audio headers
# Stage B (runtime): lean image with only what's needed to run
# ─────────────────────────────────────────────────────────────

# ── Stage A: Builder ──────────────────────────────────────────
FROM python:3.10-slim AS builder

# System audio libraries needed to compile PyAudio / sounddevice
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1-dev \
    ffmpeg \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python deps into a prefix we can copy to runtime
COPY requirements.docker.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install -r requirements.docker.txt


# ── Stage B: Runtime ──────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Only runtime audio libs (no -dev headers needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2 \
    libsndfile1 \
    ffmpeg \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY .env .

# Pre-download HF models at build time so container starts fast
# Models cached to /app/models (mounted as volume in compose)
ENV HF_CACHE_DIR=/app/models
ENV WHISPER_CACHE=/app/models/whisper
ENV TRANSFORMERS_CACHE=/app/models/transformers

# Download Whisper tiny + CodeT5 small at build time
RUN python3 -c "\
import whisper; \
whisper.load_model('tiny', download_root='/app/models/whisper'); \
print('Whisper downloaded')"

RUN python3 -c "\
from transformers import AutoTokenizer, T5ForConditionalGeneration; \
AutoTokenizer.from_pretrained('Salesforce/codet5-small', cache_dir='/app/models/transformers'); \
T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small', cache_dir='/app/models/transformers'); \
print('CodeT5 downloaded')"

# Expose MLflow UI port (sidecar in compose)
EXPOSE 5000

# Default: run system check so we can verify container health
CMD ["python", "main.py", "--check"]
