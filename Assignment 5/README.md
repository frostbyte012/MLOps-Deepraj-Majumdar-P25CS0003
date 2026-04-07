# 🚀 DLOps Assignment 5 — LoRA Fine-tuning & Adversarial Attacks

**Name:** Deepraj Majumdar  
**Roll No:** P25CS0003  

### 🔗 Quick Links
> **GitHub Repository:** [Assignment Branch](https://github.com/frostbyte012/MLOps-Deepraj-Majumdar-P25CS0003/tree/Assignment_4)  
> **WandB Q1 (ViT-S + LoRA):** [DLOps-Ass5-Q1](https://wandb.ai/tech-frostbyte-dev-012/DLOps-Ass5-Q1?nw=nwusertechfrostbytedev)  
> **WandB Q2 (Adversarial):** [DLOps-Ass5-Q2](https://wandb.ai/tech-frostbyte-dev-012/DLOps-Ass5-Q2?nw=nwusertechfrostbytedev)  
> **HuggingFace Model:** [frostbyte012/vit-s-lora-cifar100](https://huggingface.co/frostbyte012/vit-s-lora-cifar100)

---

## 📁 Repository Structure
```text
Assignment-5/
├── Q1/
│   └── train_vit_lora.py       # ViT-S + LoRA on CIFAR-100 (Grid Search & Optuna)
├── Q2/
│   ├── fgsm_attack.py          # FGSM from scratch vs IBM ART on ResNet18
│   └── adversarial_detection.py# PGD/BIM adversarial detector (ResNet34)
├── results/                    # Auto-created: checkpoints, plots, JSONs
├── Dockerfile                  # Containerized environment setup
├── requirements.txt            # Python dependencies
├── run_all.sh                  # Master execution script (Automates Q1 & Q2)
└── README.md                   # You are here!
````

-----

## ⚙️ How It Works & How to Run

This project is completely automated and containerized using Docker. The easiest way to execute the entire pipeline (CIFAR downloading, ViT-S fine-tuning, LoRA grid search, Optuna sweeping, HF pushing, and Adversarial Attack/Detection) is to use the provided bash script.

### 🐳 The 1-Click Docker Method (Recommended)

Make sure Docker is installed and your NVIDIA runtime is configured.

```bash
# Make the script executable
chmod +x run_all.sh

# Run the master pipeline
./run_all.sh
```

### 🛠️ Manual / Local Execution

If you prefer to run the files individually outside of the master script:

```bash
# 1. Setup Virtual Environment
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Login to WandB and HuggingFace
wandb login
huggingface-cli login

# 3. Run Q1 (ViT-S + LoRA Grid Search)
python Q1/train_vit_lora.py --mode all --epochs 10 --lr 1e-4 --save_dir ./results/Q1

# 4. Run Q2i (Victim ResNet18 + FGSM Attack)
python Q2/fgsm_attack.py --epochs 50 --lr 0.1 --save_dir ./results/Q2i

# 5. Run Q2ii (Adversarial Detector)
python Q2/adversarial_detection.py --victim_ckpt ./results/Q2i/resnet18_clean.pt --epochs 15 --eps 0.03 --save_dir ./results/Q2ii
```

-----

## 📊 Q1 Results: ViT-S Fine-tuning with LoRA

By injecting LoRA into the Attention layers (Q, K, V) of a pre-trained ViT-S model, we achieved a massive \~10% accuracy boost on CIFAR-100 while training a fraction of the parameters. Optuna confirmed **Rank 8 / Alpha 8** as the optimal configuration.

| LORA layers (with/without) | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Params |
|----------------------------|------|-------|---------|-----------------------|------------------|
| without                    | —    | —     | —       | 77.8%                 | \~ 38,400         |
| with                       | 2    | 2     | 0.1     | 87.4%                 | 93,796           |
| with                       | 2    | 4     | 0.1     | 87.6%                 | 93,796           |
| with                       | 2    | 8     | 0.1     | 87.2%                 | 93,796           |
| with                       | 4    | 2     | 0.1     | 87.4%                 | 149,092          |
| with                       | 4    | 4     | 0.1     | 87.5%                 | 149,092          |
| with                       | 4    | 8     | 0.1     | 87.8%                 | 149,092          |
| with                       | 8    | 2     | 0.1     | 87.3%                 | 259,684          |
| with                       | 8    | 4     | 0.1     | 87.5%                 | 259,684          |
| **with** | **8**| **8** | **0.1** | **87.9%** | **259,684** |

### Performance Comparison

-----

## 🛡️ Q2 Results: Adversarial Attacks & Detection

### Q2(i) FGSM Attack: Scratch vs IBM ART

A non-pretrained ResNet-18 was trained on CIFAR-10, achieving an **86.92%** clean accuracy. We then attacked it using FGSM. The custom Scratch implementation caused a steeper drop in accuracy at higher epsilon values compared to the ART estimator.

| ε (Epsilon) | Clean Acc | FGSM Scratch | FGSM ART | Drop (Scratch) | Drop (ART) |
|-------------|-----------|--------------|----------|----------------|------------|
| **0.000** | 86.92%    | 86.92%       | 86.92%   | 0.00%          | 0.00%      |
| **0.010** | 86.92%    | 76.25%       | 81.78%   | 10.67%         | 5.14%      |
| **0.030** | 86.92%    | 53.65%       | 61.35%   | 33.27%         | 25.57%     |
| **0.050** | 86.92%    | 36.60%       | 44.57%   | 50.32%         | 42.35%     |
| **0.100** | 86.92%    | 14.88%       | 22.79%   | 72.04%         | 64.13%     |

### Perturbation Strength Impact

-----

### Q2(ii) Adversarial Detection (ResNet-34)

A ResNet-34 binary classifier was trained to distinguish between clean images and adversarial examples generated via PGD and BIM. The detector successfully surpassed the 70% accuracy requirement.

| Attack Type | Val Accuracy | Test Det. Accuracy | ROC AUC | PR AUC |
|-------------|--------------|--------------------|---------|--------|
| **PGD** | 70.25%       | 70.80%             | 0.798   | 0.792  |
| **BIM** | 68.10%       | 70.15%             | 0.794   | 0.778  |

### PGD vs BIM Detection Performance

### Clean vs. Adversarial Visual Samples
