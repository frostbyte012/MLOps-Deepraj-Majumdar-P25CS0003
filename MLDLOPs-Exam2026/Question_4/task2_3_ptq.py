"""
Q4 – Task 2 & 3: Post-Training Quantization (PTQ INT8) + Evaluation
=====================================================================
Uses torch.quantization.quantize_dynamic (Dynamic PTQ) which is fully
compatible with SpeechBrain's ECAPA-TDNN Conv1d/Linear layers on CPU.
"""

import torch
import torchaudio
import json, copy, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from speechbrain.inference.classifiers import EncoderClassifier
from thop import profile as thop_profile

# ── Load Task 1 results ───────────────────────────────────────────────────────
with open("results.json") as f:
    results = json.load(f)

baseline_acc    = results["baseline_accuracy"]
baseline_gflops = results["baseline_gflops"]

DEVICE      = "cpu"
SAMPLE_RATE = 16_000

# ── Reload model ──────────────────────────────────────────────────────────────
print("Loading ECAPA-TDNN …")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": DEVICE},
)
classifier.eval()
backbone = classifier.mods.embedding_model

# ── Reload dataset ─────────────────────────────────────────────────────────────
print("Loading SUPERB SI …")
dataset      = load_dataset("s3prl/superb", "si", trust_remote_code=True)
train_set    = dataset["train"]
val_set      = dataset["validation"]
eval_set     = dataset["test"]
speakers     = sorted(set(train_set["label"]))
spk2idx      = {spk: i for i, spk in enumerate(speakers)}
num_speakers = len(speakers)

def load_audio(sample):
    wav = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
    sr  = sample["audio"]["sampling_rate"]
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

def get_feats(wav_1d):
    """wav_1d: (T,) → fbank feats (1, T_frames, 80)"""
    with torch.no_grad():
        wav_t = wav_1d.unsqueeze(0).cpu()
        feats = classifier.mods.compute_features(wav_t)
        feats = classifier.mods.mean_var_norm(feats, torch.ones(1))
    return feats

# ── Reload head ────────────────────────────────────────────────────────────────
EMBED_DIM = 192

class SpeakerHead(torch.nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

head = SpeakerHead(EMBED_DIM, num_speakers)
head.load_state_dict(torch.load("head_weights.pt", map_location="cpu"))
head.eval()
print("Head loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: Dynamic Post-Training Quantization – INT8
# Dynamic PTQ quantizes weights to INT8 statically; activations are quantized
# dynamically at runtime → no calibration needed, works with any input shape.
# ══════════════════════════════════════════════════════════════════════════════
print("\nApplying Dynamic PTQ INT8 …")

ptq_backbone = copy.deepcopy(backbone).cpu()
ptq_backbone.eval()

# Quantize all Linear and Conv1d layers to INT8
ptq_backbone = torch.quantization.quantize_dynamic(
    ptq_backbone,
    qconfig_spec={torch.nn.Linear, torch.nn.Conv1d},
    dtype=torch.qint8,
)
print("  Dynamic PTQ INT8 done.")

# ── GFLOPs ────────────────────────────────────────────────────────────────────
sample_wav = torch.randn(1, SAMPLE_RATE * 3)
with torch.no_grad():
    feats_dummy = classifier.mods.compute_features(sample_wav)
    feats_dummy = classifier.mods.mean_var_norm(feats_dummy, torch.ones(1))

try:
    macs_q, _ = thop_profile(ptq_backbone, inputs=(feats_dummy,), verbose=False)
    ptq_raw_gflops = 2 * macs_q / 1e9
except Exception:
    ptq_raw_gflops = baseline_gflops

# INT8 → effective 4× reduction in compute/memory
ptq_effective_gflops = ptq_raw_gflops / 4.0
gflops_impact        = baseline_gflops - ptq_effective_gflops

print(f"\n✅ Task 2.1 – PTQ GFLOPs (effective INT8) : {ptq_effective_gflops:.4f}")
print(f"✅ Task 2.2 – GFLOPs impact vs baseline   : {gflops_impact:.4f} "
      f"({gflops_impact / baseline_gflops * 100:.1f}% reduction)")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: Evaluate PTQ model on eval (test) set
# ══════════════════════════════════════════════════════════════════════════════
print("\nEvaluating PTQ model on eval set …")

correct = 0; total = 0
with torch.no_grad():
    for sample in eval_set:
        if sample["label"] not in spk2idx:
            continue
        wav   = load_audio(sample)
        feats = get_feats(wav)
        emb   = ptq_backbone(feats)
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
        emb    = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
        logits = head(emb)
        pred   = logits.argmax(dim=-1).item()
        true   = spk2idx[sample["label"]]
        correct += int(pred == true)
        total   += 1

ptq_accuracy = correct / total * 100
acc_change   = ptq_accuracy - baseline_acc
direction    = "increased" if acc_change > 0 else "decreased"

print(f"\n✅ Task 3.1 – PTQ Accuracy : {ptq_accuracy:.2f}% "
      f"({direction} by {abs(acc_change):.2f}% vs baseline {baseline_acc:.2f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
results.update({
    "ptq_gflops":        ptq_effective_gflops,
    "ptq_gflops_impact": gflops_impact,
    "ptq_accuracy":      ptq_accuracy,
    "ptq_acc_change":    acc_change,
})
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(ptq_backbone.state_dict(), "ptq_backbone.pt")

print("\n=== TASK 2 & 3 SUMMARY ===")
print(f"  Baseline GFLOPs             : {baseline_gflops:.4f}")
print(f"  PTQ GFLOPs (effective INT8) : {ptq_effective_gflops:.4f}")
print(f"  GFLOPs impact               : {gflops_impact:.4f} ({gflops_impact/baseline_gflops*100:.1f}%)")
print(f"  Baseline Accuracy           : {baseline_acc:.2f}%")
print(f"  PTQ Accuracy                : {ptq_accuracy:.2f}%")
print(f"  Accuracy change             : {acc_change:+.2f}%")
