#!/bin/bash
# =============================================================================
# run_pipeline.sh — Full MLOps Assignment 3 Pipeline
# Steps: install deps → train → push to HF Hub → evaluate from Hub
# =============================================================================

set -e  # Exit immediately on any error

# ── Configuration — EDIT THESE BEFORE RUNNING ────────────────────────────────
HF_TOKEN="YOUR_HF_TOKEN"
HF_USERNAME="YOUR_HF_USERNAME"
MODEL_NAME="distilbert-goodreads-genre"
HUB_MODEL_ID="${HF_USERNAME}/${MODEL_NAME}"

EPOCHS=10
BATCH_SIZE=10
# ─────────────────────────────────────────────────────────────────────────────

# Text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_step() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}============================================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# ── Validate config ───────────────────────────────────────────────────────────
if [[ "$HF_TOKEN" == "hf_YOURTOKEN" ]]; then
    print_error "Please edit run_pipeline.sh and set your HF_TOKEN before running."
    exit 1
fi

if [[ "$HF_USERNAME" == "YOUR_HF_USERNAME" ]]; then
    print_error "Please edit run_pipeline.sh and set your HF_USERNAME before running."
    exit 1
fi

# ── Step 1: Install dependencies ─────────────────────────────────────────────
print_step "Step 1/4 — Installing Python dependencies"
pip install -r requirements.txt
echo "Dependencies installed successfully."

# ── Step 2: Train model ───────────────────────────────────────────────────────
print_step "Step 2/4 — Training DistilBERT (epochs=${EPOCHS}, batch_size=${BATCH_SIZE})"
python src/train.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"
echo "Training complete. Model saved to ./saved_model"

# ── Step 3: Push model to HuggingFace Hub ────────────────────────────────────
print_step "Step 3/4 — Pushing model to HuggingFace Hub (${HUB_MODEL_ID})"
python src/train.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --push_to_hub \
    --hf_token "$HF_TOKEN" \
    --hub_model_id "$HUB_MODEL_ID"
echo "Model pushed to: https://huggingface.co/${HUB_MODEL_ID}"

# ── Step 4: Evaluate from HuggingFace Hub ────────────────────────────────────
print_step "Step 4/4 — Evaluating model from HuggingFace Hub"

# Wait a few seconds to allow the Hub to register the upload
echo "Waiting 10 seconds for Hub to register the model..."
sleep 10

python src/eval.py \
    --model_path "$HUB_MODEL_ID" \
    --from_hub \
    --output results/hub_eval_results.json
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