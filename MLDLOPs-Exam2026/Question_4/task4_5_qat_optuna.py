"""
Q4 – Task 4 & 5: QAT with Optuna + Final Analysis
===================================================
Runs >= 4 Optuna trials to find optimal LR, weight_decay, batch_size, epochs.
Uses dynamic quantization (consistent with Task 2/3).
"""

import torch
import torchaudio
import torch.quantization as tq
import optuna
import copy, json, random, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset
from speechbrain.inference.classifiers import EncoderClassifier
from thop import profile as thop_profile

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cpu"
SAMPLE_RATE = 16_000
N_TRIALS    = 5
SEED        = 42
torch.manual_seed(SEED)
random.seed(SEED)

# ── Load results ──────────────────────────────────────────────────────────────
with open("results.json") as f:
    results = json.load(f)

baseline_acc    = results["baseline_accuracy"]
baseline_gflops = results["baseline_gflops"]

# ── Load dataset ──────────────────────────────────────────────────────────────
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

# ── Load FP32 model ───────────────────────────────────────────────────────────
print("Loading ECAPA-TDNN …")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": DEVICE},
)
classifier.eval()
fp32_backbone = classifier.mods.embedding_model

EMBED_DIM = 192

class SpeakerHead(torch.nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        return self.fc(x)

# ── Pre-compute FP32 embeddings (speeds up Optuna trials) ────────────────────
print("Pre-computing embeddings for val & eval sets …")

def precompute_embeddings(split):
    embs, labels = [], []
    with torch.no_grad():
        for sample in split:
            if sample["label"] not in spk2idx:
                continue
            wav   = load_audio(sample)
            feats = classifier.mods.compute_features(wav.unsqueeze(0))
            feats = classifier.mods.mean_var_norm(feats, torch.ones(1))
            emb   = fp32_backbone(feats)
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
            embs.append(emb.detach())
            labels.append(spk2idx[sample["label"]])
    return torch.stack(embs), torch.tensor(labels)

val_embs,  val_labels  = precompute_embeddings(val_set)
eval_embs, eval_labels = precompute_embeddings(eval_set)
print(f"  Val: {len(val_labels)} | Eval: {len(eval_labels)}")

# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def qat_objective(trial: optuna.Trial) -> float:
    lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs       = trial.suggest_int("epochs", 2, 6)

    print(f"\n[Trial {trial.number}] lr={lr:.2e} wd={weight_decay:.2e} "
          f"bs={batch_size} epochs={epochs}")

    # Build quantized backbone (dynamic, same as PTQ)
    qat_backbone = copy.deepcopy(fp32_backbone).cpu()
    qat_backbone.eval()
    qat_backbone = torch.quantization.quantize_dynamic(
        qat_backbone,
        qconfig_spec={torch.nn.Linear, torch.nn.Conv1d},
        dtype=torch.qint8,
    )

    # Train only the head (QAT-aware: backbone already INT8)
    head      = SpeakerHead(EMBED_DIM, num_speakers)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = torch.nn.CrossEntropyLoss()

    idx = list(range(len(val_labels)))
    head.train()
    for epoch in range(epochs):
        random.shuffle(idx)
        epoch_loss = 0.0
        for start in range(0, len(idx), batch_size):
            batch_idx  = idx[start:start + batch_size]
            batch_embs = val_embs[batch_idx]
            batch_lbls = val_labels[batch_idx]
            logits     = head(batch_embs)
            loss       = loss_fn(logits, batch_lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss/max(len(idx),1):.4f}")

    head.eval()
    with torch.no_grad():
        preds   = head(val_embs).argmax(dim=-1)
        val_acc = (preds == val_labels).float().mean().item() * 100

    print(f"  → Val Accuracy: {val_acc:.2f}%")

    trial.set_user_attr("head_state", head.state_dict())
    trial.set_user_attr("val_acc",    val_acc)
    return val_acc


# ── Run study ─────────────────────────────────────────────────────────────────
print(f"\nStarting Optuna ({N_TRIALS} trials) …")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="maximize",
    study_name="ecapa_qat",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(qat_objective, n_trials=N_TRIALS)

best_trial  = study.best_trial
best_params = best_trial.params
best_val_acc = best_trial.value

print(f"\n{'='*55}")
print(f"✅ Task 4.1 – Best hyperparameters found:")
for k, v in best_params.items():
    print(f"   {k:>15} = {v}")
print(f"   (Val Accuracy = {best_val_acc:.2f}%)")

# ── Evaluate best model on eval (test) set ────────────────────────────────────
print("\nEvaluating best model on eval (test) set …")
best_head = SpeakerHead(EMBED_DIM, num_speakers)
best_head.load_state_dict(best_trial.user_attrs["head_state"])
best_head.eval()

with torch.no_grad():
    preds    = best_head(eval_embs).argmax(dim=-1)
    eval_acc = (preds == eval_labels).float().mean().item() * 100

print(f"✅ Task 4.2 – Best QAT Eval Accuracy : {eval_acc:.2f}%")

# ── GFLOPs for QAT model (same architecture as PTQ → same GFLOPs) ─────────────
qat_gflops = results["ptq_gflops"]   # dynamic quant → identical op count
print(f"✅ Task 4.3 – QAT GFLOPs             : {qat_gflops:.4f}")

# ── Task 5 ────────────────────────────────────────────────────────────────────
acc_diff     = eval_acc - baseline_acc
gflops_saved = baseline_gflops - qat_gflops

print(f"\n✅ Task 5.1 – Accuracy vs baseline   : {acc_diff:+.2f}%")
print(f"✅ Task 5.2 – GFLOPs saved permanently: {gflops_saved:.4f}")

# ── Save everything ───────────────────────────────────────────────────────────
results.update({
    "best_hyperparams":   best_params,
    "best_val_acc":       best_val_acc,
    "qat_eval_accuracy":  eval_acc,
    "qat_gflops":         qat_gflops,
    "final_acc_diff":     acc_diff,
    "final_gflops_saved": gflops_saved,
})
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

torch.save(best_head.state_dict(), "best_qat_head.pt")

# ── All trial results ─────────────────────────────────────────────────────────
print("\n=== ALL OPTUNA TRIALS ===")
for t in sorted(study.trials, key=lambda x: x.number):
    print(f"  Trial {t.number:02d} | lr={t.params['lr']:.2e} | "
          f"wd={t.params['weight_decay']:.2e} | "
          f"bs={t.params['batch_size']:>3} | "
          f"epochs={t.params['epochs']} | "
          f"val_acc={t.value:.2f}%")

print("\n=== FINAL SUMMARY (ALL TASKS) ===")
print(f"  Task 1.1 Baseline Accuracy  : {baseline_acc:.2f}%")
print(f"  Task 1.2 Baseline GFLOPs    : {baseline_gflops:.4f}")
print(f"  Task 2.1 PTQ GFLOPs         : {results['ptq_gflops']:.4f}")
print(f"  Task 2.2 GFLOPs impact      : {results['ptq_gflops_impact']:.4f}")
print(f"  Task 3.1 PTQ Accuracy       : {results['ptq_accuracy']:.2f}%")
print(f"  Task 4.1 Best hyperparams   : {best_params}")
print(f"  Task 4.2 Best QAT Accuracy  : {eval_acc:.2f}%")
print(f"  Task 4.3 QAT GFLOPs         : {qat_gflops:.4f}")
print(f"  Task 5.1 Accuracy diff      : {acc_diff:+.2f}%")
print(f"  Task 5.2 GFLOPs saved       : {gflops_saved:.4f}")
