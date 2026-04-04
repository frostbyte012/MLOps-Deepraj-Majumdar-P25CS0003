#!/bin/bash
# ==============================================================================
# Assignment 5 — Full Pipeline Runner
# Supports resuming: completed steps are skipped automatically.
# To re-run a specific step, delete its .done file from ./results/
# ==============================================================================

WANDB_KEY="6af40924706cd809ec6e12d7fbcbf9c61d7cd6d8"
HF_TOKEN="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"
HF_REPO="frostbyte012/vit-s-lora-cifar100"
CPU_ONLY=true

# ── Setup ─────────────────────────────────────────────────────────────────────
GPU_FLAG="--gpus all"
if [ "$CPU_ONLY" = true ] || ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    GPU_FLAG=""
    echo "⚠️  No NVIDIA runtime detected — running CPU-only mode."
fi

mkdir -p "$(pwd)/results"
DONE_DIR="$(pwd)/results/.done"
mkdir -p "$DONE_DIR"

# Helper: check if a step is already done
done_file() { echo "$DONE_DIR/$1.done"; }
is_done()   { [ -f "$(done_file "$1")" ]; }
mark_done() { touch "$(done_file "$1")"; echo "✅ Step '$1' marked complete."; }

# ── Step 0: Build Docker image (auto-rebuilds when Dockerfile/requirements change)
# Hash Dockerfile + requirements.txt. If either changed since last build,
# we rmi the old image and rebuild so the new deps are actually installed.
HASH_FILE="$DONE_DIR/image.hash"
CURRENT_HASH=$(md5sum Dockerfile requirements.txt 2>/dev/null | md5sum | awk '{print $1}')
STORED_HASH=$(cat "$HASH_FILE" 2>/dev/null || echo "none")

if ! docker image inspect dlops-ass5 &>/dev/null; then
    echo ""
    echo "🔨 Building Docker image (first time — this takes ~5 min)..."
    docker build -t dlops-ass5 . || { echo "❌ Docker build failed."; exit 1; }
    echo "$CURRENT_HASH" > "$HASH_FILE"
elif [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
    echo ""
    echo "🔄 Dockerfile or requirements.txt changed — rebuilding image..."
    docker rmi dlops-ass5 2>/dev/null
    docker build -t dlops-ass5 . || { echo "❌ Docker build failed."; exit 1; }
    echo "$CURRENT_HASH" > "$HASH_FILE"
else
    echo "🐳 Docker image up to date — skipping build."
    echo "   To force rebuild: rm $HASH_FILE && ./my_run.sh"
fi

COMMON_FLAGS="$GPU_FLAG --ipc=host -e WANDB_API_KEY=$WANDB_KEY -v $(pwd)/results:/workspace/results"

# ── Step 1: Q1 — ViT-S + LoRA Training ───────────────────────────────────────
echo ""
if is_done "q1_training"; then
    echo "⏭️  Q1 (ViT-S + LoRA) already done — skipping."
else
    echo "📈 Running Q1: ViT-S + LoRA Training (all combos, 10 epochs each)..."
    echo "   ⏱  Estimated time: ~5 hrs CPU / ~45 min GPU"
    docker run $COMMON_FLAGS dlops-ass5 \
        python Q1/train_vit_lora.py --mode all --epochs 10
    if [ $? -eq 0 ]; then
        mark_done "q1_training"
    else
        echo "❌ Q1 failed. Fix errors above, then re-run this script."
        exit 1
    fi
fi

# ── Step 2: Q2i — ResNet18 Training + FGSM Attack ────────────────────────────
echo ""
if is_done "q2i_fgsm"; then
    echo "⏭️  Q2i (ResNet18 + FGSM) already done — skipping."
else
    echo "🛡️  Running Q2i: ResNet18 + FGSM Attack (50 epochs)..."
    echo "   ⏱  Estimated time: ~2 hrs CPU / ~15 min GPU"
    docker run $COMMON_FLAGS dlops-ass5 \
        python Q2/fgsm_attack.py --epochs 50
    if [ $? -eq 0 ]; then
        mark_done "q2i_fgsm"
    else
        echo "❌ Q2i failed. Fix errors above, then re-run this script."
        exit 1
    fi
fi

# ── Step 3: Q2ii — Adversarial Detection ─────────────────────────────────────
echo ""
if is_done "q2ii_detection"; then
    echo "⏭️  Q2ii (Adversarial Detection) already done — skipping."
else
    # Sanity-check: victim weights must exist from Q2i
    if [ ! -f "$(pwd)/results/Q2i/resnet18_clean.pt" ]; then
        echo "❌ Victim checkpoint not found at ./results/Q2i/resnet18_clean.pt"
        echo "   Q2i must complete successfully first."
        exit 1
    fi
    echo "🔍 Running Q2ii: Adversarial Detection (PGD + BIM detectors)..."
    echo "   ⏱  Estimated time: ~3 hrs CPU / ~25 min GPU"
    docker run $COMMON_FLAGS dlops-ass5 \
        python Q2/adversarial_detection.py \
        --victim_ckpt ./results/Q2i/resnet18_clean.pt
    if [ $? -eq 0 ]; then
        mark_done "q2ii_detection"
    else
        echo "❌ Q2ii failed. Fix errors above, then re-run this script."
        exit 1
    fi
fi

# ── Step 4: Push best LoRA model to HuggingFace ──────────────────────────────
echo ""
if is_done "push_to_hub"; then
    echo "⏭️  HuggingFace push already done — skipping."
else
    BEST_CKPT="$(pwd)/results/Q1/lora_r8_a8/lora_r8_a8_best.pt"
    if [ ! -f "$BEST_CKPT" ]; then
        echo "⚠️  Best checkpoint not found at $BEST_CKPT — skipping HuggingFace push."
        echo "   Q1 must complete successfully first."
    else
        echo "🤗 Pushing best LoRA model to HuggingFace: $HF_REPO ..."
        docker run $GPU_FLAG --ipc=host \
            -e HF_TOKEN=$HF_TOKEN \
            -v "$(pwd)/results:/workspace/results" \
            dlops-ass5 \
            python push_to_hub.py \
            --ckpt ./results/Q1/lora_r8_a8/lora_r8_a8_best.pt \
            --repo "$HF_REPO"
        if [ $? -eq 0 ]; then
            mark_done "push_to_hub"
        else
            echo "❌ HuggingFace push failed — check your HF_TOKEN and repo name."
            # Non-fatal: don't exit, results are already saved locally
        fi
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "✅ All tasks completed! Results are in: $(pwd)/results/"
echo ""
echo "   Step status (delete .done file to re-run a step):"
for step in q1_training q2i_fgsm q2ii_detection push_to_hub; do
    if is_done "$step"; then
        echo "   ✅ $step"
    else
        echo "   ⬜ $step (not completed)"
    fi
done
echo ""
echo "   To force re-run a specific step, e.g. Q2i:"
echo "   rm $(done_file q2i_fgsm) && ./my_run.sh"
echo "============================================================"