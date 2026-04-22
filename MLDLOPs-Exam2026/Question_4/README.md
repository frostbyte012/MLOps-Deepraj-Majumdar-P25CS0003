# Q4: Model Optimization and Quantization for Speaker Verification

## Overview
This repository contains the solution for Q4 — profiling, quantizing, and applying Quantization-Aware Training (QAT) on the **ECAPA-TDNN** speaker verification model using the SpeechBrain VoxCeleb recipe and the SUPERB SI dataset.

---

## Setup

```bash
pip install speechbrain torch torchaudio datasets optuna thop
python solution.py
```

---

## Results

| Task | Metric | Value |
|------|--------|-------|
| **Task 1a** | Baseline ECAPA-TDNN Accuracy | `__.__%` |
| **Task 1b** | Baseline GFLOPs | `_.____` |
| **Task 2a** | PTQ (INT8) GFLOPs | `_.____` |
| **Task 2b** | GFLOPs impact vs baseline | `_.____` |
| **Task 3a** | PTQ Accuracy (increase/decrease to) | `__.__%` |
| **Task 4a** | Best Optuna hyperparameters | `lr=_, wd=_, bs=_` |
| **Task 4b** | Best QAT recovered Accuracy | `__.__%` |
| **Task 4c** | QAT single inference GFLOPs | `_.____` |
| **Task 5a** | Final absolute Accuracy difference vs baseline | `__.__%` |
| **Task 5b** | GFLOPs permanently saved | `_.____` |

> ⚠️ Fill in the values above after running `solution.py`. The script prints a ready-to-copy table at the end.

---

## Approach

### Task 1 — Baseline Profiling
- Loaded `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN backbone).
- Used `thop` to measure GFLOPs on a 3-second dummy waveform.
- Evaluated Top-1 Speaker Identification Accuracy on the SUPERB SI **Eval** split.

### Task 2 — Post-Training Quantization (PTQ)
- Applied `torch.quantization.quantize_dynamic` with `dtype=torch.qint8` to all `nn.Linear` and `nn.Conv1d` layers.
- Re-profiled GFLOPs on the quantized graph.

### Task 3 — Initial Comparative Analysis
- Evaluated PTQ model on the SUPERB SI Eval split.
- Reported absolute Accuracy change and GFLOPs difference vs baseline.

### Task 4 — QAT + Optuna
- Inserted fake-quantize observers (`get_default_qat_qconfig("fbgemm")`).
- Fine-tuned on the SUPERB SI **Val** split.
- Used **Optuna** (≥ 4 trials) to search over:
  - `lr` ∈ [1e-5, 1e-3] (log-uniform)
  - `weight_decay` ∈ [1e-6, 1e-2] (log-uniform)
  - `batch_size` ∈ {4, 8, 16}
- Re-trained best configuration and evaluated on the Eval split.

### Task 5 — Trade-off Evaluation
- Compared best QAT model against baseline on Accuracy and GFLOPs.

---

## Repository Structure

```
.
├── solution.py      # Main script (Tasks 1-5)
└── README.md        # This file
```