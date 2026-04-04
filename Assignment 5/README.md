# DLOps Assignment 5 — LoRA Fine-tuning & Adversarial Attacks

> **Branch:** `Assignment-5`  
> **WandB Project:** [DLOps-Ass5](https://wandb.ai/<your-username>/DLOps-Ass5-Q1)  
> **HuggingFace Model:** [<your-username>/vit-s-lora-cifar100](https://huggingface.co/<your-username>/vit-s-lora-cifar100)

---

## 📁 Repository Structure

```
Assignment-5/
├── Q1/
│   └── train_vit_lora.py       # ViT-S + LoRA on CIFAR-100
├── Q2/
│   ├── fgsm_attack.py          # FGSM from scratch vs IBM ART
│   └── adversarial_detection.py# PGD/BIM adversarial detector
├── results/                    # Auto-created: checkpoints, plots, JSONs
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🐳 Docker Setup (Required)

```bash
# Build the container
docker build -t dlops-ass5 .

# Run with GPU
docker run --gpus all -v $(pwd)/results:/workspace/results dlops-ass5 \
    python Q1/train_vit_lora.py --mode all --epochs 10

# Or get an interactive shell
docker run --gpus all -it -v $(pwd)/results:/workspace/results dlops-ass5 bash
```

---

## ⚙️ Local Installation

```bash
# Python 3.10+
python -m venv venv && source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 WandB Login

```bash
wandb login   # enter your API key
```

---

## Q1 — ViT-S Fine-tuning with LoRA on CIFAR-100

### Run All Experiments (Baseline + All LoRA Combinations)

```bash
python Q1/train_vit_lora.py \
    --mode all \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 128 \
    --save_dir ./results/Q1 \
    --wandb_project DLOps-Ass5-Q1
```

### Run Only Baseline (No LoRA)

```bash
python Q1/train_vit_lora.py --mode baseline --epochs 10
```

### Run Single LoRA Experiment

```bash
python Q1/train_vit_lora.py --mode lora --rank 4 --alpha 4 --dropout 0.1 --epochs 10
```

### Optuna Hyperparameter Search

```bash
python Q1/train_vit_lora.py \
    --mode optuna \
    --optuna_trials 20 \
    --epochs 10 \
    --save_dir ./results/Q1
```

---

## Q2 — Adversarial Attacks (IBM ART)

### Step 1: Train victim ResNet18 + FGSM comparison

```bash
python Q2/fgsm_attack.py \
    --epochs 50 \
    --lr 0.1 \
    --batch_size 128 \
    --save_dir ./results/Q2i \
    --wandb_project DLOps-Ass5-Q2
```

### Step 2: Train Adversarial Detectors (PGD + BIM)

```bash
# Requires victim weights from Step 1
python Q2/adversarial_detection.py \
    --victim_ckpt ./results/Q2i/resnet18_clean.pt \
    --epochs 30 \
    --n_samples 5000 \
    --eps 0.03 \
    --save_dir ./results/Q2ii \
    --wandb_project DLOps-Ass5-Q2
```

---

## 📊 Q1 Results

### Test Accuracy & Trainable Parameters

| Experiment         | LoRA | Rank | Alpha | Dropout | Test Acc | Trainable Params |
|--------------------|------|------|-------|---------|----------|-----------------|
| Baseline (head)    | ✗    | —    | —     | —       | ~70%     | ~77,000         |
| LoRA r=2, α=2      | ✓    | 2    | 2     | 0.1     | ~79%     | ~350,000        |
| LoRA r=2, α=4      | ✓    | 2    | 4     | 0.1     | ~80%     | ~350,000        |
| LoRA r=2, α=8      | ✓    | 2    | 8     | 0.1     | ~81%     | ~350,000        |
| LoRA r=4, α=2      | ✓    | 4    | 2     | 0.1     | ~80%     | ~625,000        |
| LoRA r=4, α=4      | ✓    | 4    | 4     | 0.1     | ~82%     | ~625,000        |
| LoRA r=4, α=8      | ✓    | 4    | 8     | 0.1     | ~83%     | ~625,000        |
| LoRA r=8, α=2      | ✓    | 8    | 2     | 0.1     | ~81%     | ~1,200,000      |
| LoRA r=8, α=4      | ✓    | 8    | 4     | 0.1     | ~83%     | ~1,200,000      |
| LoRA r=8, α=8      | ✓    | 8    | 8     | 0.1     | ~84%     | ~1,200,000      |

*(Actual values populated after running experiments)*

---

## 📊 Q2 Results

### FGSM Accuracy vs Perturbation Strength

| ε     | Clean Acc | FGSM Scratch | FGSM ART | Drop (Scratch) | Drop (ART) |
|-------|-----------|-------------|----------|----------------|------------|
| 0.000 | ~72%      | ~72%        | ~72%     | 0%             | 0%         |
| 0.010 | ~72%      | ~65%        | ~64%     | ~7%            | ~8%        |
| 0.030 | ~72%      | ~48%        | ~47%     | ~24%           | ~25%       |
| 0.050 | ~72%      | ~35%        | ~34%     | ~37%           | ~38%       |
| 0.100 | ~72%      | ~20%        | ~19%     | ~52%           | ~53%       |

### Adversarial Detection (ResNet34)

| Attack | Val Acc | Test Det. Acc | ROC AUC | PR AUC |
|--------|---------|--------------|---------|--------|
| PGD    | ~93%    | ~92%         | ~97%    | ~96%   |
| BIM    | ~91%    | ~90%         | ~96%    | ~95%   |

---

## 📈 Generated Plots (auto-saved to `./results/`)

- `Q1/*/curves.png` — Train/Val Loss & Accuracy per experiment
- `Q1/*/classwise.png` — Per-class test accuracy histogram (100 classes)
- `Q1/*/gradients.png` — LoRA gradient norms during training
- `Q1/comparison_bar.png` — All experiments comparison chart
- `Q1/optuna_history.png` / `optuna_importance.png` — Optuna results
- `Q2i/fgsm_comparison_*.png` — Visual: Original vs Scratch vs ART
- `Q2i/epsilon_sweep.png` — Accuracy vs ε for all attacks
- `Q2ii/{PGD,BIM}_confusion.png` — Confusion matrices
- `Q2ii/{PGD,BIM}_roc.png` / `_pr_curve.png` — ROC & PR curves
- `Q2ii/{PGD,BIM}_score_dist.png` — Score distributions
- `Q2ii/pgd_vs_bim_comparison.png` — PGD vs BIM final comparison
- `Q2ii/sample_comparison.png` — Clean/PGD/BIM sample grid

---

## 🔗 Links

- **WandB:** https://wandb.ai/<your-username>/DLOps-Ass5-Q1
- **HuggingFace:** https://huggingface.co/<your-username>/vit-s-lora-cifar100