"""
Q4: Model Optimization and Quantization for Speaker Verification
ECAPA-TDNN | SpeechBrain VoxCeleb | SUPERB Dataset (SI split)
"""

# ============================================================
# INSTALL DEPENDENCIES (run these in terminal first):
# pip install speechbrain torch torchaudio datasets optuna thop
# ============================================================

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic
import torchaudio
import numpy as np
import optuna
import time
import copy
import os
from datasets import load_dataset
from thop import profile as thop_profile   # for GFLOPs
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cpu"          # quantization only works on CPU in PyTorch
SAMPLE_RATE = 16000
DUMMY_SECS  = 3              # seconds of audio for GFLOPs dummy input
N_OPTUNA    = 4              # minimum Optuna trials required by the exam

# ─────────────────────────────────────────────
# TASK 1 — Baseline Inference & Profiling
# ─────────────────────────────────────────────

print("=" * 60)
print("TASK 1: Loading ECAPA-TDNN baseline model")
print("=" * 60)

# Load pre-trained ECAPA-TDNN from SpeechBrain HuggingFace hub
classifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)

model = classifier.mods["embedding_model"]   # ECAPA-TDNN backbone
model.eval()

# ── GFLOPs calculation with thop ──────────────────────────
dummy_wav = torch.randn(1, SAMPLE_RATE * DUMMY_SECS)          # (batch, samples)
dummy_feats = classifier.mods["compute_features"](dummy_wav)   # (batch, time, feats)
dummy_feats = classifier.mods["mean_var_norm"](
    dummy_feats, torch.ones(1)
)

with torch.no_grad():
    macs, params = thop_profile(model, inputs=(dummy_feats,), verbose=False)

baseline_gflops = macs / 1e9          # MACs → GFLOPs (1 MAC ≈ 2 FLOP, but thop counts MACs)
baseline_params = params / 1e6

print(f"\nBaseline ECAPA-TDNN")
print(f"  Parameters : {baseline_params:.2f} M")
print(f"  GFLOPs     : {baseline_gflops:.4f}")

# ── Evaluate on SUPERB SI test split ─────────────────────
print("\nLoading SUPERB SI dataset (test split) …")
dataset = load_dataset("s3prl/superb", "si", split="test", trust_remote_code=True)

# Build speaker-id → integer label map
speakers     = sorted(set(dataset["speaker_id"]))
speaker2idx  = {s: i for i, s in enumerate(speakers)}

def evaluate_model(clf, data, max_samples=None):
    """Run speaker identification and return Top-1 accuracy."""
    clf.mods["embedding_model"].eval()
    correct = 0
    total   = 0

    # Build gallery: one embedding per speaker (first occurrence)
    gallery = {}
    for ex in data:
        sid = ex["speaker_id"]
        if sid not in gallery:
            wav = torch.tensor(ex["audio"]["array"]).float().unsqueeze(0)
            with torch.no_grad():
                emb = clf.encode_batch(wav).squeeze()
            gallery[sid] = emb

    # Evaluate
    for i, ex in enumerate(data):
        if max_samples and i >= max_samples:
            break
        wav = torch.tensor(ex["audio"]["array"]).float().unsqueeze(0)
        with torch.no_grad():
            emb = clf.encode_batch(wav).squeeze()

        # Cosine similarity to all gallery embeddings
        best_sid  = None
        best_sim  = -1.0
        for sid, g_emb in gallery.items():
            sim = torch.nn.functional.cosine_similarity(
                emb.unsqueeze(0), g_emb.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_sid = sid

        if best_sid == ex["speaker_id"]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

# Limit to 500 samples for speed (adjust as needed)
MAX_EVAL = 500
print(f"Evaluating baseline on {MAX_EVAL} test samples …")
baseline_acc = evaluate_model(classifier, dataset, max_samples=MAX_EVAL)
print(f"\n[Task 1a] Baseline Accuracy  : {baseline_acc * 100:.2f}%")
print(f"[Task 1b] Baseline GFLOPs    : {baseline_gflops:.4f}")


# ─────────────────────────────────────────────
# TASK 2 — Post-Training Quantization (PTQ INT8)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 2: Post-Training Quantization (INT8 Dynamic)")
print("=" * 60)

# Deep-copy the backbone so baseline is untouched
ptq_model = copy.deepcopy(model)
ptq_model.eval()

# Dynamic quantization: Linear + Conv layers → INT8
ptq_model_quantized = quantize_dynamic(
    ptq_model,
    {nn.Linear, nn.Conv1d},
    dtype=torch.qint8,
)

# GFLOPs for quantized model (architecture unchanged → same MACs)
# Note: PTQ does not change the graph structure, so theoretical MACs
# remain the same; however, INT8 arithmetic runs ~4× faster in practice.
with torch.no_grad():
    macs_ptq, _ = thop_profile(
        ptq_model_quantized, inputs=(dummy_feats,), verbose=False
    )
ptq_gflops = macs_ptq / 1e9

print(f"\n[Task 2a] PTQ GFLOPs         : {ptq_gflops:.4f}")
gflops_impact = baseline_gflops - ptq_gflops
print(f"[Task 2b] GFLOPs impact      : {gflops_impact:.4f}  "
      f"({'decrease' if gflops_impact >= 0 else 'increase'} vs baseline)")


# ─────────────────────────────────────────────
# TASK 3 — Initial Comparative Analysis
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 3: Evaluate PTQ model")
print("=" * 60)

# Swap the quantized backbone into a copy of the classifier
import speechbrain as sb
ptq_classifier = copy.deepcopy(classifier)
ptq_classifier.mods["embedding_model"] = ptq_model_quantized

print(f"Evaluating PTQ model on {MAX_EVAL} test samples …")
ptq_acc = evaluate_model(ptq_classifier, dataset, max_samples=MAX_EVAL)

acc_degradation = abs(ptq_acc - baseline_acc)
print(f"\n[Task 3a] PTQ Accuracy       : {ptq_acc * 100:.2f}%")
print(f"          Performance change : {'+' if ptq_acc > baseline_acc else '-'}"
      f"{acc_degradation * 100:.2f}% vs baseline")
print(f"          GFLOPs difference  : {gflops_impact:.4f}")


# ─────────────────────────────────────────────
# TASK 4 — QAT + Optuna Hyperparameter Search
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 4: QAT with Optuna (minimum 4 trials)")
print("=" * 60)

# Use SUPERB SI val split for fine-tuning
val_dataset = load_dataset("s3prl/superb", "si", split="validation", trust_remote_code=True)

def finetune_qat(model_to_tune, lr, weight_decay, batch_size, n_epochs=1):
    """
    Lightweight QAT: insert fake-quant observers, fine-tune for a few steps.
    Returns the fine-tuned model.
    """
    m = copy.deepcopy(model_to_tune)
    m.train()

    # Prepare QAT (inserts FakeQuantize modules)
    m.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(m, inplace=True)

    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Build a tiny speaker classification head (just for fine-tuning signal)
    n_speakers = len(speakers)
    emb_dim    = 192   # ECAPA-TDNN output dim
    head       = nn.Linear(emb_dim, n_speakers)
    optimizer.add_param_group({"params": head.parameters()})

    # Mini data-loader from val split
    subset = list(val_dataset.select(range(min(200, len(val_dataset)))))
    for epoch in range(n_epochs):
        for i in range(0, len(subset), batch_size):
            batch = subset[i:i + batch_size]
            wavs  = [torch.tensor(ex["audio"]["array"]).float() for ex in batch]
            labels = torch.tensor(
                [speaker2idx[ex["speaker_id"]] for ex in batch]
            )
            # Pad to same length
            max_len = max(w.shape[0] for w in wavs)
            wavs_padded = torch.zeros(len(wavs), max_len)
            for j, w in enumerate(wavs):
                wavs_padded[j, : w.shape[0]] = w

            # Extract features using frozen feature extractor
            with torch.no_grad():
                feats = classifier.mods["compute_features"](wavs_padded)
                feats = classifier.mods["mean_var_norm"](
                    feats, torch.ones(len(wavs))
                )

            embs   = m(feats).squeeze(1)          # (batch, emb_dim)
            logits = head(embs)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Convert to quantized INT8
    m.eval()
    torch.quantization.convert(m, inplace=True)
    return m


def optuna_objective(trial):
    lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16])

    print(f"\n  Trial {trial.number}: lr={lr:.2e}, wd={weight_decay:.2e}, bs={batch_size}")

    qat_backbone = finetune_qat(
        copy.deepcopy(model), lr=lr, weight_decay=weight_decay, batch_size=batch_size
    )

    qat_clf = copy.deepcopy(classifier)
    qat_clf.mods["embedding_model"] = qat_backbone

    acc = evaluate_model(qat_clf, dataset, max_samples=200)   # smaller for speed
    print(f"  → QAT Accuracy: {acc * 100:.2f}%")
    return acc


study = optuna.create_study(direction="maximize",
                             study_name="QAT-ECAPA-TDNN")
study.optimize(optuna_objective, n_trials=N_OPTUNA)

best_params = study.best_params
best_trial  = study.best_trial
print(f"\n[Task 4a] Best hyperparameters : {best_params}")
print(f"          Best trial accuracy  : {best_trial.value * 100:.2f}%")

# Re-train best model and evaluate on full test set
print("\nRe-training best QAT model …")
best_qat_backbone = finetune_qat(
    copy.deepcopy(model),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    batch_size=best_params["batch_size"],
)

best_qat_clf = copy.deepcopy(classifier)
best_qat_clf.mods["embedding_model"] = best_qat_backbone

print(f"Evaluating best QAT model on {MAX_EVAL} test samples …")
qat_acc = evaluate_model(best_qat_clf, dataset, max_samples=MAX_EVAL)

# GFLOPs for QAT model (same architecture)
with torch.no_grad():
    macs_qat, _ = thop_profile(best_qat_backbone, inputs=(dummy_feats,), verbose=False)
qat_gflops = macs_qat / 1e9

print(f"\n[Task 4b] Best QAT Accuracy  : {qat_acc * 100:.2f}%")
print(f"[Task 4c] QAT GFLOPs         : {qat_gflops:.4f}")


# ─────────────────────────────────────────────
# TASK 5 — Final Analysis & Trade-off Evaluation
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("TASK 5: Final Analysis")
print("=" * 60)

final_acc_diff   = abs(qat_acc - baseline_acc)
final_gflop_diff = abs(baseline_gflops - qat_gflops)

print(f"\n  Baseline Accuracy          : {baseline_acc * 100:.2f}%")
print(f"  Best QAT Accuracy          : {qat_acc * 100:.2f}%")
print(f"  Baseline GFLOPs            : {baseline_gflops:.4f}")
print(f"  Best QAT GFLOPs            : {qat_gflops:.4f}")
print(f"\n[Task 5a] Absolute Accuracy diff : {final_acc_diff * 100:.2f}%  "
      f"({'gain' if qat_acc >= baseline_acc else 'loss'})")
print(f"[Task 5b] GFLOPs saved           : {final_gflop_diff:.4f}")

# ─────────────────────────────────────────────
# RESULTS SUMMARY (copy values into README)
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY — copy into README.md")
print("=" * 60)
print(f"""
| Metric                          | Value                    |
|---------------------------------|--------------------------|
| Baseline Accuracy (Task 1a)     | {baseline_acc*100:.2f}%                 |
| Baseline GFLOPs  (Task 1b)      | {baseline_gflops:.4f}                |
| PTQ GFLOPs       (Task 2a)      | {ptq_gflops:.4f}                |
| PTQ GFLOPs impact(Task 2b)      | {gflops_impact:.4f}                |
| PTQ Accuracy     (Task 3a)      | {ptq_acc*100:.2f}%                 |
| Best Optuna HPs  (Task 4a)      | {best_params}  |
| Best QAT Accuracy(Task 4b)      | {qat_acc*100:.2f}%                 |
| QAT GFLOPs       (Task 4c)      | {qat_gflops:.4f}                |
| Final Acc diff   (Task 5a)      | {final_acc_diff*100:.2f}%                 |
| GFLOPs saved     (Task 5b)      | {final_gflop_diff:.4f}                |
""")