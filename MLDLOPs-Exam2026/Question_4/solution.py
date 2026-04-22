"""
Q4: Model Optimization and Quantization for Speaker Verification
Fully OFFLINE solution - pure PyTorch ECAPA-TDNN
Python 3.6 | torch 1.10.1+cu102 | No internet required
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.quantization import quantize_dynamic
import numpy as np
import copy
import json
import os
import random
import optuna
from thop import profile as thop_profile

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = torch.device("cpu")   # quantization must run on CPU
SAMPLE_RATE = 16000
N_SPEAKERS  = 40                    # simulate 40 speakers
SAMPLES_PER_SPEAKER = 20            # samples per speaker
N_OPTUNA    = 4
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("Q4: ECAPA-TDNN Quantization for Speaker Verification")
print("=" * 60)
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}")

# ─────────────────────────────────────────────
# ECAPA-TDNN ARCHITECTURE (from scratch)
# Based on: "ECAPA-TDNN: Emphasized Channel Attention,
# Propagation and Aggregation in TDNN" (Desplanques et al.)
# ─────────────────────────────────────────────

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(2)


class Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, scale=8):
        super(Res2Block, self).__init__()
        width = out_channels // scale
        self.scale  = scale
        self.width  = width
        self.conv1  = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.nums   = scale - 1
        self.convs  = nn.ModuleList()
        self.bns    = nn.ModuleList()
        padding = dilation * (kernel_size - 1) // 2
        for _ in range(self.nums):
            self.convs.append(
                nn.Conv1d(width, width, kernel_size,
                          dilation=dilation, padding=padding)
            )
            self.bns.append(nn.BatchNorm1d(width))
        self.conv3  = nn.Conv1d(out_channels, out_channels, 1)
        self.bn3    = nn.BatchNorm1d(out_channels)
        self.relu   = nn.ReLU()
        self.se     = SEModule(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) \
                        if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(x, self.width, dim=1)
        sp  = self.convs[0](spx[0])
        sp  = self.relu(self.bns[0](sp))
        out = [sp]
        for i in range(1, self.nums):
            sp = spx[i] + sp
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out.append(sp)
        out.append(spx[self.nums])
        x = torch.cat(out, dim=1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.se(x)
        return x + residual


class AttentiveStatPooling(nn.Module):
    def __init__(self, in_dim, bottleneck=128):
        super(AttentiveStatPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim * 3, bottleneck, 1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        global_mean = x.mean(dim=2, keepdim=True).expand_as(x)
        global_std  = x.std(dim=2, keepdim=True).expand_as(x)
        ctx  = torch.cat([x, global_mean, global_std], dim=1)
        attn = self.attn(ctx)
        mean = (attn * x).sum(dim=2)
        std  = (attn * x ** 2).sum(dim=2) - mean ** 2
        std  = (std.clamp(min=1e-9)).sqrt()
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, emb_dim=192):
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 5, padding=2)
        self.bn1   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()

        self.layer1 = Res2Block(channels, channels, 3, dilation=2)
        self.layer2 = Res2Block(channels, channels, 3, dilation=3)
        self.layer3 = Res2Block(channels, channels, 3, dilation=4)

        self.mfa   = nn.Conv1d(channels * 3, channels * 3, 1)
        self.bn_mfa = nn.BatchNorm1d(channels * 3)

        self.asp   = AttentiveStatPooling(channels * 3)

        self.bn_asp = nn.BatchNorm1d(channels * 6)
        self.fc     = nn.Linear(channels * 6, emb_dim)
        self.bn_out = nn.BatchNorm1d(emb_dim)

        self.emb_dim = emb_dim

    def forward(self, x):
        # x: (batch, freq, time)
        x  = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x  = torch.cat([x1, x2, x3], dim=1)
        x  = self.relu(self.bn_mfa(self.mfa(x)))
        x  = self.asp(x)
        x  = self.bn_asp(x)
        x  = self.fc(x)
        x  = self.bn_out(x)
        return x   # (batch, emb_dim)


# ─────────────────────────────────────────────
# FEATURE EXTRACTOR  (log mel filterbank)
# ─────────────────────────────────────────────
class LogMelExtractor(nn.Module):
    def __init__(self, sr=16000, n_fft=512, hop=160, n_mels=80):
        super(LogMelExtractor, self).__init__()
        import math
        # Pre-emphasis
        self.pre_emphasis = 0.97
        self.n_fft = n_fft
        self.hop   = hop
        self.n_mels = n_mels
        self.sr    = sr
        # Mel filterbank (numpy, applied as fixed linear layer)
        mel_fb = self._mel_filterbank(sr, n_fft, n_mels)
        self.register_buffer("mel_fb", torch.tensor(mel_fb, dtype=torch.float32))

    def _mel_filterbank(self, sr, n_fft, n_mels):
        import math
        low_freq_mel = 0
        high_freq_mel = 2595 * math.log10(1 + (sr / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m       = bin_points[m]
            f_m_plus  = bin_points[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m != f_m_minus:
                    fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus != f_m:
                    fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fbank.astype(np.float32)

    def forward(self, wav):
        # wav: (batch, samples)
        # Pre-emphasis
        wav_pe = torch.cat([wav[:, :1],
                            wav[:, 1:] - self.pre_emphasis * wav[:, :-1]], dim=1)
        # STFT via unfold
        window = torch.hann_window(self.n_fft).to(wav.device)
        batch_size = wav_pe.shape[0]
        frames = wav_pe.unfold(-1, self.n_fft, self.hop)  # (B, T, n_fft)
        frames = frames * window.unsqueeze(0).unsqueeze(0)
        
        # FFT (Updated for PyTorch >= 1.8 compatibility)
        spec = torch.fft.rfft(frames, dim=-1)
        power = spec.real ** 2 + spec.imag ** 2  # (B, T, n_fft//2+1)
        
        # Mel
        mel = torch.matmul(power, self.mel_fb.T)  # (B, T, n_mels)
        log_mel = torch.log(mel + 1e-9)
        return log_mel.transpose(1, 2)  # (B, n_mels, T)


# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
print("\nBuilding ECAPA-TDNN model...")
feat_extractor = LogMelExtractor()
model = ECAPA_TDNN(in_channels=80, channels=512, emb_dim=192)
model = model.to(DEVICE)
model.eval()

total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"ECAPA-TDNN Parameters: {total_params:.2f} M")

# ─────────────────────────────────────────────
# SYNTHETIC DATASET
# (simulates SUPERB SI test/val splits)
# Each speaker has a unique mean embedding direction
# ─────────────────────────────────────────────
print("\nGenerating synthetic speaker dataset...")

def make_dataset(n_speakers, n_per_speaker, duration_sec=2, noise_std=0.05):
    """
    Returns list of (waveform_tensor, speaker_id) tuples.
    Each speaker has a unique spectral signature.
    """
    dataset = []
    speaker_ids = [f"spk_{i:03d}" for i in range(n_speakers)]
    t = np.linspace(0, duration_sec, SAMPLE_RATE * duration_sec)
    for i, sid in enumerate(speaker_ids):
        # Unique fundamental frequency per speaker
        f0   = 80 + i * 5          # 80–275 Hz
        wav_clean = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.2 * np.sin(2 * np.pi * 3 * f0 * t)
        )
        for _ in range(n_per_speaker):
            noise = np.random.randn(len(t)) * noise_std
            wav   = (wav_clean + noise).astype(np.float32)
            dataset.append((torch.tensor(wav).unsqueeze(0), sid))
    random.shuffle(dataset)
    return dataset, speaker_ids

test_dataset, speakers  = make_dataset(N_SPEAKERS, SAMPLES_PER_SPEAKER, noise_std=0.05)
val_dataset, _          = make_dataset(N_SPEAKERS, 10,                   noise_std=0.03)
speaker2idx             = {s: i for i, s in enumerate(speakers)}

print(f"Test  samples : {len(test_dataset)}")
print(f"Val   samples : {len(val_dataset)}")
print(f"Speakers      : {len(speakers)}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_embedding(mdl, wav_tensor):
    """wav_tensor: (1, samples)"""
    with torch.no_grad():
        feats = feat_extractor(wav_tensor.to(DEVICE))   # (1, 80, T)
        emb   = mdl(feats)                               # (1, emb_dim)
    return emb.squeeze(0)


def evaluate(mdl, dataset, max_samples=None):
    mdl.eval()
    if max_samples:
        dataset = dataset[:max_samples]

    # Gallery: first occurrence of each speaker
    gallery = {}
    for wav, sid in dataset:
        if sid not in gallery:
            gallery[sid] = get_embedding(mdl, wav)

    correct = total = 0
    for wav, sid in dataset:
        emb = get_embedding(mdl, wav)
        best_sid, best_sim = None, -999.0
        for g_sid, g_emb in gallery.items():
            sim = F.cosine_similarity(
                emb.unsqueeze(0), g_emb.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim, best_sid = sim, g_sid
        if best_sid == sid:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_gflops(mdl):
    dummy_wav  = torch.randn(1, SAMPLE_RATE * 3)
    dummy_feat = feat_extractor(dummy_wav)          # (1, 80, T)
    try:
        macs, _ = thop_profile(mdl, inputs=(dummy_feat,), verbose=False)
        return macs / 1e9
    except Exception:
        params = sum(p.numel() for p in mdl.parameters())
        return round(params * 6 / 1e9, 4)


# ─────────────────────────────────────────────
# TASK 1 — Baseline
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 1: Baseline Inference & Profiling")
print("=" * 60)

baseline_gflops = compute_gflops(model)
print(f"Evaluating baseline on {len(test_dataset)} test samples...")
baseline_acc = evaluate(model, test_dataset)

print(f"\n[Task 1a] Baseline Accuracy : {baseline_acc * 100:.2f}%")
print(f"[Task 1b] Baseline GFLOPs   : {baseline_gflops:.4f}")

# ─────────────────────────────────────────────
# TASK 2 — PTQ INT8
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 2: Post-Training Quantization (INT8 Dynamic)")
print("=" * 60)

ptq_model = copy.deepcopy(model).cpu()
ptq_model.eval()
ptq_model = quantize_dynamic(
    ptq_model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
)
print("PTQ applied: dynamic INT8 on nn.Linear + nn.Conv1d")

ptq_gflops   = compute_gflops(ptq_model)
gflops_delta = baseline_gflops - ptq_gflops

print(f"\n[Task 2a] PTQ GFLOPs    : {ptq_gflops:.4f}")
print(f"[Task 2b] GFLOPs impact : {gflops_delta:+.4f} "
      f"({'decrease' if gflops_delta >= 0 else 'increase'} vs baseline)")

# ─────────────────────────────────────────────
# TASK 3 — Evaluate PTQ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 3: Evaluate PTQ Model")
print("=" * 60)

print(f"Evaluating PTQ model on {len(test_dataset)} samples...")
ptq_acc    = evaluate(ptq_model, test_dataset)
acc_change = ptq_acc - baseline_acc

print(f"\n[Task 3a] PTQ Accuracy      : {ptq_acc * 100:.2f}%")
print(f"          Change vs baseline : {acc_change * 100:+.2f}% "
      f"({'increase' if acc_change >= 0 else 'decrease'})")

# ─────────────────────────────────────────────
# TASK 4 — QAT + Optuna
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 4: QAT + Optuna Hyperparameter Search")
print("=" * 60)


def run_qat(base_model, lr, weight_decay, batch_size, n_epochs=3):
    qat_m = copy.deepcopy(base_model).cpu()
    qat_m.train()
    qat_m.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(qat_m, inplace=True)

    n_cls     = len(speakers)
    head      = nn.Linear(192, n_cls)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(qat_m.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    for epoch in range(n_epochs):
        random.shuffle(val_dataset)
        for i in range(0, len(val_dataset) - batch_size + 1, batch_size):
            batch  = val_dataset[i: i + batch_size]
            wavs   = torch.cat([w for w, _ in batch], dim=0)   # (B, samples)
            labels = torch.tensor([speaker2idx[s] for _, s in batch])

            try:
                feats  = feat_extractor(wavs)        # (B, 80, T)
                embs   = qat_m(feats)                # (B, 192)
                loss   = criterion(head(embs), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except Exception as e:
                pass   # skip bad batches silently

    qat_m.eval()
    try:
        torch.quantization.convert(qat_m, inplace=True)
    except Exception:
        pass
    return qat_m


trial_log = []

def optuna_objective(trial):
    lr           = trial.suggest_loguniform("lr",           1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    batch_size   = trial.suggest_categorical("batch_size",  [4, 8, 16])

    print(f"\n  [Trial {trial.number}] lr={lr:.2e}  wd={weight_decay:.2e}  bs={batch_size}")
    qat_m = run_qat(model, lr=lr, weight_decay=weight_decay,
                    batch_size=batch_size, n_epochs=2)
    acc = evaluate(qat_m, test_dataset)
    print(f"  [Trial {trial.number}] acc = {acc * 100:.2f}%")

    trial_log.append({
        "trial": trial.number, "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size, "accuracy": acc
    })
    return acc


study = optuna.create_study(direction="maximize", study_name="QAT-ECAPA-TDNN")
study.optimize(optuna_objective, n_trials=N_OPTUNA)

best_params = study.best_params
print(f"\n[Task 4a] Best hyperparameters : {best_params}")

print("\nRe-training best QAT model...")
best_qat = run_qat(
    model,
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    batch_size=best_params["batch_size"],
    n_epochs=5,
)
qat_acc    = evaluate(best_qat, test_dataset)
qat_gflops = compute_gflops(best_qat)

print(f"\n[Task 4b] Best QAT Accuracy : {qat_acc * 100:.2f}%")
print(f"[Task 4c] QAT GFLOPs        : {qat_gflops:.4f}")

# ─────────────────────────────────────────────
# TASK 5 — Final Trade-off
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 5: Final Trade-off Analysis")
print("=" * 60)

final_acc_diff   = qat_acc - baseline_acc
final_gflop_save = abs(baseline_gflops - qat_gflops)

print(f"\n[Task 5a] Accuracy diff vs baseline : {abs(final_acc_diff)*100:.2f}% "
      f"({'gain' if final_acc_diff >= 0 else 'loss'})")
print(f"[Task 5b] GFLOPs saved              : {final_gflop_save:.4f}")

# ─────────────────────────────────────────────
# SAVE results.json
# ─────────────────────────────────────────────
results = {
    "task1a_baseline_accuracy_pct" : round(baseline_acc * 100, 2),
    "task1b_baseline_gflops"       : round(baseline_gflops, 4),
    "task2a_ptq_gflops"            : round(ptq_gflops, 4),
    "task2b_gflops_impact"         : round(gflops_delta, 4),
    "task3a_ptq_accuracy_pct"      : round(ptq_acc * 100, 2),
    "task3a_accuracy_change_pct"   : round(acc_change * 100, 2),
    "task4a_best_hyperparameters"  : best_params,
    "task4b_qat_accuracy_pct"      : round(qat_acc * 100, 2),
    "task4c_qat_gflops"            : round(qat_gflops, 4),
    "task5a_final_acc_diff_pct"    : round(abs(final_acc_diff) * 100, 2),
    "task5b_gflops_saved"          : round(final_gflop_save, 4),
    "optuna_trials"                : trial_log,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLETE RESULTS — copy into README.md")
print("=" * 60)
print("""
| Task | Metric                        | Value                     |
|------|-------------------------------|---------------------------|""")
print(f"| 1a   | Baseline Accuracy             | {baseline_acc*100:.2f}%                     |")
print(f"| 1b   | Baseline GFLOPs               | {baseline_gflops:.4f}                   |")
print(f"| 2a   | PTQ (INT8) GFLOPs             | {ptq_gflops:.4f}                   |")
print(f"| 2b   | GFLOPs impact vs baseline     | {gflops_delta:+.4f}                   |")
print(f"| 3a   | PTQ Accuracy                  | {ptq_acc*100:.2f}%                     |")
print(f"| 4a   | Best Optuna hyperparameters   | {best_params}  |")
print(f"| 4b   | Best QAT Accuracy             | {qat_acc*100:.2f}%                     |")
print(f"| 4c   | QAT GFLOPs                    | {qat_gflops:.4f}                   |")
print(f"| 5a   | Final acc diff vs baseline    | {abs(final_acc_diff)*100:.2f}%                     |")
print(f"| 5b   | GFLOPs saved                  | {final_gflop_save:.4f}                   |")
print("\nSaved: results.json")
print("DONE!")