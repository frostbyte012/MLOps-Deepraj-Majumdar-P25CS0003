#!/bin/bash
# =============================================================================
# run_pipeline.sh — Full MLOps Assignment 3 Pipeline
#
# Flow:
#   Step 1 — Train locally with GPU (faster than Docker on this machine)
#   Step 2 — Push trained model to HuggingFace Hub
#   Step 3 — Build eval Docker image (Task 9)
#   Step 4 — Run eval container (pulls model from HF Hub + evaluates)
# =============================================================================

set -e

# ── Configuration — EDIT THESE BEFORE RUNNING ────────────────────────────────
HF_TOKEN="YOUR_HF_TOKEN"
HF_USERNAME="YOUR_HF_USERNAME"
MODEL_NAME="distilbert-goodreads-genre"
HUB_MODEL_ID="${HF_USERNAME}/${MODEL_NAME}"

EPOCHS=10
BATCH_SIZE=10
EVAL_IMAGE="goodreads-eval"
# ─────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}============================================================${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# ── Validate config ───────────────────────────────────────────────────────────
if [[ "$HF_TOKEN" == "hf_YOURTOKEN" ]]; then
    print_error "Please set your HF_TOKEN in run_pipeline.sh before running."
    exit 1
fi
if [[ "$HF_USERNAME" == "YOUR_HF_USERNAME" ]]; then
    print_error "Please set your HF_USERNAME in run_pipeline.sh before running."
    exit 1
fi

# ── Check Docker ──────────────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH."
    exit 1
fi
if ! docker info &> /dev/null; then
    print_error "Cannot connect to Docker daemon. Run: newgrp docker"
    exit 1
fi

# ── Check GPU ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)${NC}"
    echo -e "${YELLOW}Training will run locally to use GPU directly.${NC}"
    echo -e "${YELLOW}(nvidia-container-toolkit not available on this machine for Docker GPU passthrough)${NC}"
else
    echo -e "${YELLOW}No GPU detected — training will run on CPU.${NC}"
fi

mkdir -p results logs saved_model

# ── Step 1: Install dependencies & train locally with GPU ────────────────────
print_step "Step 1/4 — Installing dependencies"
pip install -r requirements.txt --quiet

print_step "Step 2/4 — Training locally with GPU + pushing to HuggingFace Hub"
python src/train.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --push_to_hub \
    --hf_token "$HF_TOKEN" \
    --hub_model_id "$HUB_MODEL_ID"

echo "Training complete. Model pushed to: https://huggingface.co/${HUB_MODEL_ID}"

# ── Step 3: Build eval Docker image (Task 9) ──────────────────────────────────
print_step "Step 3/4 — Building evaluation Docker image: ${EVAL_IMAGE}"
docker build \
    --build-arg HF_MODEL_ID="$HUB_MODEL_ID" \
    -t "$EVAL_IMAGE" \
    -f Dockerfile.eval .
echo "Eval image built successfully."

# ── Step 4: Run eval container (pulls model from HF Hub) ─────────────────────
print_step "Step 4/4 — Running eval container (Task 9 — pulls model from HuggingFace Hub)"

echo "Waiting 15 seconds for HuggingFace Hub to register the model..."
sleep 15

RESULTS_DIR="$(pwd)/results"
docker run \
    --rm \
    -e HF_MODEL_ID="$HUB_MODEL_ID" \
    -e WANDB_DISABLED="true" \
    -v "${RESULTS_DIR}:/app/results" \
    "$EVAL_IMAGE"

echo "Hub evaluation results saved to: results/hub_eval_results.json"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Pipeline complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Local eval results : results/local_eval_results.json"
echo "  Hub eval results   : results/hub_eval_results.json"
echo "  HuggingFace model  : https://huggingface.co/${HUB_MODEL_ID}"
echo ""