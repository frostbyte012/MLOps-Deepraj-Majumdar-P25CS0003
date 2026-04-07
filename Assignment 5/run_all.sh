#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Assignment 5 — One-command runner
#  Just run: ./run_all.sh
#
#  Auto-detects what is already done by checking real output files.
#  Safe to Ctrl+C and re-run — continues from where it stopped.
# ═══════════════════════════════════════════════════════════════════
set -e

# ── API Keys ───────────────────────────────────────────────────────
WANDB_API_KEY=""
HF_TOKEN=""
HF_REPO=""
# ──────────────────────────────────────────────────────────────────

IMAGE_NAME="dlops-ass5"
RESULTS_DIR="$(pwd)/results"
DATA_DIR="$(pwd)/data"
mkdir -p "$RESULTS_DIR" "$DATA_DIR"

# ── Fix permissions on data/ so Docker never blocks downloads ──────
sudo chown -R $USER:$USER "$DATA_DIR" 2>/dev/null || true
chmod -R 755 "$DATA_DIR" 2>/dev/null || true

# ── Pre-download datasets if not already present ───────────────────
if [ ! -d "$DATA_DIR/cifar-100-python" ]; then
    echo "📦 Downloading CIFAR-100 (161MB)..."
    rm -f "$DATA_DIR/cifar-100-python.tar.gz"
    wget -q --show-progress \
        https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz \
        -O "$DATA_DIR/cifar-100-python.tar.gz"
    tar -xzf "$DATA_DIR/cifar-100-python.tar.gz" -C "$DATA_DIR/"
    rm -f "$DATA_DIR/cifar-100-python.tar.gz"
    echo "✅ CIFAR-100 ready."
else
    echo "✅ CIFAR-100 already downloaded — skipping."
fi

if [ ! -d "$DATA_DIR/cifar-10-batches-py" ]; then
    echo "📦 Downloading CIFAR-10 (163MB)..."
    rm -f "$DATA_DIR/cifar-10-python.tar.gz"
    wget -q --show-progress \
        https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
        -O "$DATA_DIR/cifar-10-python.tar.gz"
    tar -xzf "$DATA_DIR/cifar-10-python.tar.gz" -C "$DATA_DIR/"
    rm -f "$DATA_DIR/cifar-10-python.tar.gz"
    echo "✅ CIFAR-10 ready."
else
    echo "✅ CIFAR-10 already downloaded — skipping."
fi

##USE THIS ONLY : JUST FOR THE TIME BEING I"M USING SOMETHING ELSE

# # ── Completion checkers — based on real output files ───────────────
# q1_baseline_done() { [ -f "$RESULTS_DIR/Q1/baseline/result.json" ]; }
# q1_lora_done() {
#     local count=0
#     for r in 2 4 8; do for a in 2 4 8; do
#         [ -f "$RESULTS_DIR/Q1/lora_r${r}_a${a}/result.json" ] && count=$((count+1))
#     done; done
#     [ "$count" -eq 9 ]
# }
# q1_optuna_done()  { [ -f "$RESULTS_DIR/Q1/optuna_history.png" ]; }
# q1_push_done()    { [ -f "$RESULTS_DIR/Q1/.hf_pushed" ]; }
# q2i_done()        { [ -f "$RESULTS_DIR/Q2i/epsilon_sweep.png" ]; }
# q2ii_done()       { [ -f "$RESULTS_DIR/Q2ii/pgd_vs_bim_comparison.png" ]; }

# ── Completion checkers — based on real output files ───────────────
q1_baseline_done() { return 0; }
q1_lora_done()     { return 0; }
q1_optuna_done()   { return 0; }
q1_push_done()    { [ -f "$RESULTS_DIR/Q1/.hf_pushed" ]; }
q2i_done()        { [ -f "$RESULTS_DIR/Q2i/epsilon_sweep.png" ]; }
q2ii_done()       { [ -f "$RESULTS_DIR/Q2ii/pgd_vs_bim_comparison.png" ]; }

# ── Common docker run command ──────────────────────────────────────
DRUN="docker run --gpus all --rm \
  --shm-size=8g \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -e HF_TOKEN=${HF_TOKEN} \
  -e HF_HUB_OFFLINE=1 \
  -v ${RESULTS_DIR}:/workspace/results \
  -v ${DATA_DIR}:/workspace/data \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ${IMAGE_NAME}"

# ══════════════════════════════════════════════════════════════════
#  STEP 0: Build Docker image — only if missing
# ══════════════════════════════════════════════════════════════════
echo ""
if docker image inspect $IMAGE_NAME &>/dev/null; then
    echo "🐳 Docker image exists — skipping build."
else
    echo "🔨 Building Docker image (~5 min, only happens once)..."
    docker build -t $IMAGE_NAME .
    echo "✅ Image built."
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 1: Q1 Baseline
# ══════════════════════════════════════════════════════════════════
echo ""
if q1_baseline_done; then
    echo "⏭️  Q1 Baseline done — skipping."
else
    echo "📈 Q1 [1/3]: ViT-S Baseline — head only, no LoRA (~90 min)..."
    $DRUN python3 Q1/train_vit_lora.py \
        --mode baseline --epochs 10 --lr 1e-4 --batch_size 64 \
        --save_dir /workspace/results/Q1 \
        --wandb_project DLOps-Ass5-Q1
    echo "✅ Q1 Baseline done."
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 2: Q1 LoRA (all 9 combos)
# ══════════════════════════════════════════════════════════════════
echo ""
if q1_lora_done; then
    echo "⏭️  Q1 LoRA all 9 combos done — skipping."
else
    echo "📈 Q1 [2/3]: LoRA combos r∈{2,4,8} × α∈{2,4,8} (~35 min)..."
    $DRUN python3 Q1/train_vit_lora.py \
        --mode lora --epochs 10 --lr 1e-4 --batch_size 64 \
        --save_dir /workspace/results/Q1 \
        --wandb_project DLOps-Ass5-Q1 \
        --skip_baseline --skip_done
    echo "✅ Q1 LoRA done."
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 3: Q1 Optuna HPO
# ══════════════════════════════════════════════════════════════════
echo ""
if q1_optuna_done; then
    echo "⏭️  Q1 Optuna done — skipping."
else
    echo "🔍 Q1 [3/3]: Optuna HPO — 20 trials (~30 min)..."
    $DRUN python3 Q1/train_vit_lora.py \
        --mode optuna --epochs 10 --optuna_trials 20 \
        --save_dir /workspace/results/Q1 \
        --wandb_project DLOps-Ass5-Q1
    echo "✅ Q1 Optuna done."
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 4: Push best model to HuggingFace
# ══════════════════════════════════════════════════════════════════
echo ""
if q1_push_done; then
    echo "⏭️  HuggingFace push done — skipping."
else
    echo "🤗 Pushing best LoRA model to HuggingFace..."
    BEST_CKPT=$(python3 - <<PYEOF
import json, glob
best_acc, best_ckpt = 0, ""
for f in glob.glob("results/Q1/lora_*/result.json"):
    try:
        d = json.load(open(f))
        if d.get("test_acc", 0) > best_acc:
            best_acc = d["test_acc"]
            best_ckpt = f"results/Q1/{d['name']}/{d['name']}_best.pt"
    except: pass
print(best_ckpt)
PYEOF
)
    if [ -n "$BEST_CKPT" ] && [ -f "$BEST_CKPT" ]; then
        docker run --gpus all --rm \
          -e HF_TOKEN=${HF_TOKEN} \
          -v ${RESULTS_DIR}:/workspace/results \
          ${IMAGE_NAME} \
          python3 push_to_hub.py \
            --ckpt /workspace/results/${BEST_CKPT#results/} \
            --hf_token "${HF_TOKEN}"
        touch "$RESULTS_DIR/Q1/.hf_pushed"
        echo "✅ Pushed to HuggingFace."
    else
        echo "⚠️  No LoRA checkpoint found — skipping HF push."
    fi
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 5: Q2i — ResNet18 + FGSM
# ══════════════════════════════════════════════════════════════════
echo ""
if q2i_done; then
    echo "⏭️  Q2i done — skipping."
else
    echo "⚔️  Q2i: ResNet18 CIFAR-10 + FGSM attack (~10 min)..."
    SKIP_TRAIN=""
    [ -f "$RESULTS_DIR/Q2i/resnet18_clean.pt" ] && SKIP_TRAIN="--skip_train"
    $DRUN python3 Q2/fgsm_attack.py \
        --epochs 50 --lr 0.1 --batch_size 128 \
        --save_dir /workspace/results/Q2i \
        --wandb_project DLOps-Ass5-Q2 \
        $SKIP_TRAIN
    echo "✅ Q2i done."
fi

# ══════════════════════════════════════════════════════════════════
#  STEP 6: Q2ii — Adversarial Detection
# ══════════════════════════════════════════════════════════════════
echo ""
if q2ii_done; then
    echo "⏭️  Q2ii done — skipping."
else
    if [ ! -f "$RESULTS_DIR/Q2i/resnet18_clean.pt" ]; then
        echo "❌ Victim model missing — Q2i must complete first."
        exit 1
    fi
    echo "🛡️  Q2ii: Adversarial Detectors PGD + BIM (~15 min)..."
    $DRUN python3 Q2/adversarial_detection.py \
        --victim_ckpt /workspace/results/Q2i/resnet18_clean.pt \
        --epochs 30 --n_samples 5000 --eps 0.03 \
        --save_dir /workspace/results/Q2ii \
        --wandb_project DLOps-Ass5-Q2
    echo "✅ Q2ii done."
fi

# ══════════════════════════════════════════════════════════════════
echo ""
echo "🎉 All steps complete!"
echo "   Results     → $RESULTS_DIR"
echo "   HuggingFace → https://huggingface.co/${HF_REPO}"
echo "   WandB Q1    → https://wandb.ai/m23cse012/DLOps-Ass5-Q1"
echo "   WandB Q2    → https://wandb.ai/m23cse012/DLOps-Ass5-Q2"
