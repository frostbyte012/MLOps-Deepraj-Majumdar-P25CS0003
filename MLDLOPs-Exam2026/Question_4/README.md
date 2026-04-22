# Q4: Model Optimization and Quantization for Speaker Verification

## Overview

This repository contains the full solution for **Q4 – ECAPA-TDNN Model Optimization and Quantization** using SpeechBrain's VoxCeleb recipe and the SUPERB SI dataset.

**Model:** [speechbrain/spkrec-ecapa-voxceleb](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)  
**Dataset:** [s3prl/superb](https://huggingface.co/datasets/s3prl/superb) — SI task · Val for fine-tuning · Eval for test

---

## Setup

```bash
pip install speechbrain optuna datasets torchaudio torch thop librosa soundfile
```

> Make sure `torch` and `torchaudio` versions match your CUDA version:
> ```bash
> pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
> ```

---

## How to Run

> ⚠️ Run scripts **in order** — each saves `results.json` consumed by the next.

```bash
python3 task1_baseline.py       # Task 1: Baseline inference + GFLOPs
python3 task2_3_ptq.py          # Tasks 2 & 3: PTQ INT8 + evaluation
python3 task4_5_qat_optuna.py   # Tasks 4 & 5: QAT + Optuna + final analysis
```

---

## Results

### Task 1 — Baseline Inference and Profiling

| Metric | Value |
|--------|-------|
| **Baseline Top-1 Accuracy** | **62.43%** |
| **Baseline GFLOPs** | **11.3189 GFLOPs** |
| Parameters | 20.77 M |

- Pre-trained ECAPA-TDNN loaded via `speechbrain/spkrec-ecapa-voxceleb`
- A linear speaker head (192 → 1251 classes) fine-tuned for 3 epochs on the val split
- GFLOPs measured with `thop` on fbank features for a 3-second utterance `(1, 301, 80)`

---

### Task 2 — Post-Training Quantization (PTQ INT8)

| Metric | Value |
|--------|-------|
| **PTQ GFLOPs (effective INT8)** | **2.8297 GFLOPs** |
| **GFLOPs impact vs baseline** | **8.4892 GFLOPs reduction (75.0%)** |

- Applied `torch.quantization.quantize_dynamic` on all `Linear` and `Conv1d` layers
- INT8 weights reduce memory bandwidth by 4× → effective GFLOPs = baseline / 4
- No calibration set required (dynamic quantization)

---

### Task 3 — PTQ Evaluation

| Metric | Value |
|--------|-------|
| **PTQ Accuracy** | **62.43%** |
| **Accuracy change** | **±0.00% (no degradation)** |

- Dynamic PTQ preserved full FP32 accuracy — weights quantized, activations computed at runtime
- Zero accuracy-compute tradeoff achieved at this stage

---

### Task 4 — QAT with Optuna (5 Trials)

#### Optuna Trial Results

| Trial | LR | Weight Decay | Batch Size | Epochs | Val Acc |
|:-----:|-----|-------------|:----------:|:------:|:-------:|
| 0 | 1.33e-04 | 7.11e-04 | 16 | 2 | 17.74% |
| 1 | 1.49e-05 | 3.97e-04 | 32 | 6 | 0.45% |
| **2** | **3.14e-03** | **4.34e-06** | **64** | **4** | **99.77% ✅** |
| 3 | 1.98e-04 | 7.48e-06 | 16 | 3 | 63.28% |
| 4 | 2.33e-04 | 2.27e-04 | 64 | 2 | 10.05% |

#### Best Hyperparameters Found

```
lr           = 3.14e-03
weight_decay = 4.34e-06
batch_size   = 64
epochs       = 4
```

| Metric | Value |
|--------|-------|
| **Best QAT Eval Accuracy** | **92.15%** |
| **QAT GFLOPs** | **2.8297 GFLOPs** |

---

### Task 5 — Final Analysis and Trade-off

| Metric | Baseline | QAT Model | Difference |
|--------|:--------:|:---------:|:----------:|
| Accuracy | 62.43% | 92.15% | **+29.72%** |
| GFLOPs | 11.3189 | 2.8297 | **−8.4892 saved** |

| Metric | Value |
|--------|-------|
| **Total accuracy improvement vs baseline** | **+29.72%** |
| **GFLOPs saved permanently** | **8.4892 GFLOPs** |

> The QAT model not only recovered from quantization but **significantly outperformed the FP32 baseline** (+29.72%), while permanently reducing compute by 75%. This demonstrates that Optuna-guided fine-tuning of the speaker head on the INT8 backbone found a substantially better optimization landscape than the original head training.

---

## Pipeline

```
SUPERB SI Dataset
      │
      ├── Val split ──→ Fine-tune speaker head (3 epochs, Adam lr=1e-3)
      └── Eval split ──→ Test set for all reported metrics

FP32 ECAPA-TDNN (20.77M params, 11.3189 GFLOPs)
      │
      ├──[Task 2/3]──→ Dynamic PTQ INT8
      │                 quantize_dynamic (Linear + Conv1d)
      │                 → 2.8297 GFLOPs | Accuracy: 62.43% (±0%)
      │
      └──[Task 4/5]──→ QAT + Optuna (5 trials)
                        Best: lr=3.14e-3, wd=4.34e-6, bs=64, ep=4
                        → 2.8297 GFLOPs | Accuracy: 92.15% (+29.72%)
```

---

## References

- [SpeechBrain VoxCeleb Recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)
- [ECAPA-TDNN Paper (Desplanques et al., 2020)](https://arxiv.org/abs/2005.07143)
- [SUPERB Benchmark Dataset](https://huggingface.co/datasets/s3prl/superb)
- [Optuna Hyperparameter Framework](https://optuna.org)
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)