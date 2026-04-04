"""
Assignment 5 - Q2(ii): Adversarial Detection using ResNet-34
Detects clean vs adversarial images (PGD and BIM attacks via IBM ART).
Binary classifier: clean=0, adversarial=1
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, classification_report)
import wandb
import logging

from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']


# ──────────────────────────── Data ───────────────────────────────────────────

def get_cifar10_numpy(train: bool = True):
    """Return CIFAR-10 as normalized numpy arrays."""
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR10('./data', train=train, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy()); ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


# ──────────────────────────── Victim model (ResNet18) ────────────────────────

def build_victim_resnet18(num_classes=10):
    m = models.resnet18(weights=None)              # pretrained=False is deprecated in torchvision ≥ 0.13
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def get_art_victim(model, device):
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(-3.0, 3.0),
        device_type='gpu' if torch.cuda.is_available() else 'cpu',
    )


# ──────────────────────────── Attack generation ──────────────────────────────

def generate_adversarial_samples(art_classifier, x_clean: np.ndarray,
                                  attack_type: str = 'pgd',
                                  eps: float = 0.03,
                                  n_samples: int = 5000) -> np.ndarray:
    """Generate adversarial examples via PGD or BIM."""
    x = x_clean[:n_samples]

    if attack_type == 'pgd':
        attack = ProjectedGradientDescent(
            estimator=art_classifier,
            eps=eps,
            eps_step=eps / 10,
            max_iter=40,
            targeted=False,
            batch_size=256,
        )
    elif attack_type == 'bim':
        attack = BasicIterativeMethod(
            estimator=art_classifier,
            eps=eps,
            eps_step=eps / 10,
            max_iter=40,
            targeted=False,
            batch_size=256,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_type}")

    logger.info(f"Generating {attack_type.upper()} adversarial samples (n={n_samples})...")
    x_adv = attack.generate(x=x)
    logger.info(f"Done. Mean perturbation: {np.abs(x_adv - x).mean():.4f}")
    return x_adv


# ──────────────────────────── Detector (ResNet34) ────────────────────────────

def build_detector_resnet34() -> nn.Module:
    """Binary classifier (clean=0, adversarial=1) using ResNet34.

    Input: 6-channel tensor [image (3ch) | high-freq residual (3ch)].
    The residual channel directly exposes the adversarial noise pattern.
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    # Patch first conv to accept 6 channels; preserve pretrained RGB weights.
    old_conv = model.conv1
    new_conv = nn.Conv2d(6, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight  # keep RGB weights
        new_conv.weight[:, 3:] = 0.0               # zeros for residual
    model.conv1 = new_conv
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )
    return model


def make_detector_input(x: np.ndarray) -> np.ndarray:
    """Build 6-channel detector input: [image | high-freq residual].
    Residual = |x - uniform_blur(x)| amplifies adversarial noise patterns.
    """
    from scipy.ndimage import uniform_filter
    blurred  = uniform_filter(x, size=(1, 1, 3, 3))
    residual = np.abs(x - blurred)
    return np.concatenate([x, residual], axis=1).astype(np.float32)


def build_detector_dataset(x_clean, x_adv):
    """Create balanced 6-channel dataset of clean (0) and adversarial (1) samples."""
    n = min(len(x_clean), len(x_adv))
    x_c = make_detector_input(x_clean[:n])
    x_a = make_detector_input(x_adv[:n])
    x = np.concatenate([x_c, x_a], axis=0)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    # Shuffle
    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]

    # Train/val split (80/20)
    split = int(0.8 * len(x))
    return (x[:split], y[:split]), (x[split:], y[split:])


def train_detector(model, train_data, val_data, epochs, lr, save_path, device,
                   attack_type, wandb_run=None):
    x_tr, y_tr = train_data
    x_va, y_va = val_data

    tr_dataset = TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr))
    va_dataset = TensorDataset(torch.tensor(x_va), torch.tensor(y_va))
    tr_loader  = DataLoader(tr_dataset, batch_size=128, shuffle=True,  num_workers=4)
    va_loader  = DataLoader(va_dataset, batch_size=128, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {'tr_loss': [], 'va_loss': [], 'tr_acc': [], 'va_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)
        scheduler.step()

        # Validation
        model.eval()
        va_loss, va_correct, va_total = 0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in va_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                va_loss += loss.item() * labels.size(0)
                va_correct += (out.argmax(1) == labels).sum().item()
                va_total += labels.size(0)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        history['tr_loss'].append(tr_loss / tr_total)
        history['va_loss'].append(va_loss / va_total)
        history['tr_acc'].append(tr_acc)
        history['va_acc'].append(va_acc)

        logger.info(f"[{attack_type}] Epoch {epoch:>3} | TrAcc:{tr_acc*100:.2f}% | VaAcc:{va_acc*100:.2f}%")
        if wandb_run:
            wandb_run.log({f'{attack_type}/epoch': epoch,
                           f'{attack_type}/tr_acc': tr_acc,
                           f'{attack_type}/va_acc': va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), save_path)
            np.save(save_path.replace('.pt', '_probs.npy'), np.array(all_probs))
            np.save(save_path.replace('.pt', '_labels.npy'), np.array(all_labels))

    return best_acc, history, np.array(all_probs), np.array(all_labels)


# ──────────────────────────── Evaluation Plots ───────────────────────────────

def save_detector_plots(all_probs, all_labels, history, attack_type, save_dir, wandb_run=None):
    os.makedirs(save_dir, exist_ok=True)

    # ─── 1. Loss & Accuracy Curves ───
    epochs = range(1, len(history['tr_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['tr_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['va_loss'], 'r-o', label='Val')
    axes[0].set_title(f'{attack_type} Detector — Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, [a*100 for a in history['tr_acc']], 'b-o', label='Train')
    axes[1].plot(epochs, [a*100 for a in history['va_acc']], 'r-o', label='Val')
    axes[1].set_title(f'{attack_type} Detector — Accuracy'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    curves_path = os.path.join(save_dir, f'{attack_type}_curves.png')
    plt.savefig(curves_path, dpi=150); plt.close()

    # ─── 2. Confusion Matrix ───
    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Adversarial'],
                yticklabels=['Clean', 'Adversarial'], ax=ax)
    ax.set_title(f'{attack_type} — Confusion Matrix')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, f'{attack_type}_confusion.png')
    plt.savefig(cm_path, dpi=150); plt.close()

    # ─── 3. ROC Curve ───
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{attack_type} — ROC Curve'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, f'{attack_type}_roc.png')
    plt.savefig(roc_path, dpi=150); plt.close()

    # ─── 4. Precision-Recall Curve ───
    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, 'g-', lw=2, label=f'PR AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'{attack_type} — Precision-Recall Curve'); ax.legend(); ax.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, f'{attack_type}_pr_curve.png')
    plt.savefig(pr_path, dpi=150); plt.close()

    # ─── 5. Score Distribution ───
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_probs[all_labels == 0], bins=50, alpha=0.6, color='green', label='Clean')
    ax.hist(all_probs[all_labels == 1], bins=50, alpha=0.6, color='red',   label='Adversarial')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax.set_xlabel('Adversarial Probability Score')
    ax.set_ylabel('Count')
    ax.set_title(f'{attack_type} — Detection Score Distribution')
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    dist_path = os.path.join(save_dir, f'{attack_type}_score_dist.png')
    plt.savefig(dist_path, dpi=150); plt.close()

    if wandb_run:
        wandb_run.log({
            f'{attack_type}/confusion_matrix': wandb.Image(cm_path),
            f'{attack_type}/roc_curve':        wandb.Image(roc_path),
            f'{attack_type}/pr_curve':         wandb.Image(pr_path),
            f'{attack_type}/score_dist':       wandb.Image(dist_path),
            f'{attack_type}/curves':           wandb.Image(curves_path),
            f'{attack_type}/roc_auc':          roc_auc,
            f'{attack_type}/pr_auc':           pr_auc,
        })

    # Classification report
    print(f"\n{'='*50}")
    print(f"[{attack_type}] Classification Report:")
    print(classification_report(all_labels, preds, target_names=['Clean', 'Adversarial']))

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}


def save_adversarial_samples_wandb(x_clean, x_adv_pgd, x_adv_bim, y_clean,
                                    save_dir, wandb_run, n=10):
    """Log 10 clean + adversarial samples for each attack to WandB."""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])

    def denorm(x):
        img = x.copy()
        for c in range(3): img[c] = img[c] * std[c] + mean[c]
        return np.clip(img, 0, 1).transpose(1, 2, 0)

    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    for i in range(n):
        axes[0][i].imshow(denorm(x_clean[i]))
        axes[0][i].set_title(CIFAR10_CLASSES[y_clean[i]], fontsize=7)
        axes[0][i].axis('off')
        axes[1][i].imshow(denorm(x_adv_pgd[i]))
        axes[1][i].set_title('PGD', fontsize=7)
        axes[1][i].axis('off')
        axes[2][i].imshow(denorm(x_adv_bim[i]))
        axes[2][i].set_title('BIM', fontsize=7)
        axes[2][i].axis('off')
    axes[0][0].set_ylabel('Clean', fontsize=9)
    axes[1][0].set_ylabel('PGD Adv.', fontsize=9)
    axes[2][0].set_ylabel('BIM Adv.', fontsize=9)
    plt.suptitle('Clean vs Adversarial Samples', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'sample_comparison.png')
    plt.savefig(path, dpi=150); plt.close()
    if wandb_run:
        wandb_run.log({'sample_comparison': wandb.Image(path)})
    return path


def save_pgd_bim_comparison(pgd_metrics, bim_metrics, save_dir, wandb_run=None):
    """Side-by-side bar comparison of PGD vs BIM detection metrics."""
    metrics = ['Detection Accuracy', 'ROC AUC', 'PR AUC']
    pgd_vals = [pgd_metrics['test_acc'] * 100, pgd_metrics['roc_auc'] * 100, pgd_metrics['pr_auc'] * 100]
    bim_vals = [bim_metrics['test_acc'] * 100, bim_metrics['roc_auc'] * 100, bim_metrics['pr_auc'] * 100]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - width/2, pgd_vals, width, label='PGD', color='steelblue', alpha=0.85)
    b2 = ax.bar(x + width/2, bim_vals, width, label='BIM', color='tomato',    alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel('Score (%)'); ax.set_ylim(0, 110)
    ax.set_title('PGD vs BIM — Adversarial Detector Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(axis='y')
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                           f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                           f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, 'pgd_vs_bim_comparison.png')
    plt.savefig(path, dpi=150); plt.close()
    if wandb_run:
        wandb_run.log({'pgd_vs_bim_comparison': wandb.Image(path)})
    return path


# ──────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--victim_ckpt', type=str, default='./results/Q2i/resnet18_clean.pt')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--lr',         type=float, default=3e-5)  # lower LR critical for stable detector training
    parser.add_argument('--n_samples',  type=int,   default=5000)
    parser.add_argument('--eps',        type=float, default=0.03)
    parser.add_argument('--save_dir',   type=str,   default='./results/Q2ii')
    parser.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    parser.add_argument('--no_wandb',   action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run = None
    if not args.no_wandb:
        run = wandb.init(project=args.wandb_project, name='adversarial-detection')

    # ── Load victim model ──
    victim = build_victim_resnet18().to(device)
    if os.path.exists(args.victim_ckpt):
        victim.load_state_dict(torch.load(args.victim_ckpt, map_location=device))
        logger.info("Loaded victim model.")
    else:
        raise FileNotFoundError(f"Victim checkpoint not found: {args.victim_ckpt}\n"
                                 "Run fgsm_attack.py first to train the victim.")
    victim.eval()
    art_victim = get_art_victim(victim, device)

    # ── Load CIFAR-10 ──
    logger.info("Loading CIFAR-10...")
    x_train, y_train = get_cifar10_numpy(train=True)
    x_test,  y_test  = get_cifar10_numpy(train=False)

    # ── Generate adversarial examples ──
    pgd_cache = os.path.join(args.save_dir, f'pgd_adv_eps{args.eps:.3f}.npy')
    bim_cache = os.path.join(args.save_dir, f'bim_adv_eps{args.eps:.3f}.npy')

    if os.path.exists(pgd_cache):
        x_train_pgd = np.load(pgd_cache)
        logger.info("Loaded cached PGD adversarials.")
    else:
        x_train_pgd = generate_adversarial_samples(art_victim, x_train, 'pgd', args.eps, args.n_samples)
        np.save(pgd_cache, x_train_pgd)

    if os.path.exists(bim_cache):
        x_train_bim = np.load(bim_cache)
        logger.info("Loaded cached BIM adversarials.")
    else:
        x_train_bim = generate_adversarial_samples(art_victim, x_train, 'bim', args.eps, args.n_samples)
        np.save(bim_cache, x_train_bim)

    # ── WandB samples ──
    x_test_pgd = generate_adversarial_samples(art_victim, x_test, 'pgd', args.eps, min(500, len(x_test)))
    x_test_bim = generate_adversarial_samples(art_victim, x_test, 'bim', args.eps, min(500, len(x_test)))
    save_adversarial_samples_wandb(x_test[:10], x_test_pgd[:10], x_test_bim[:10],
                                    y_test[:10], args.save_dir, run, n=10)

    all_metrics = {}

    for attack_type, x_adv in [('PGD', x_train_pgd), ('BIM', x_train_bim)]:
        logger.info(f"\n{'='*60}\nTraining {attack_type} Detector\n{'='*60}")

        (tr_data, va_data) = build_detector_dataset(x_train[:args.n_samples], x_adv)

        detector = build_detector_resnet34().to(device)
        ckpt = os.path.join(args.save_dir, f'detector_{attack_type.lower()}.pt')

        best_acc, history, probs, labels = train_detector(
            detector, tr_data, va_data, args.epochs, args.lr,
            ckpt, device, attack_type, run
        )

        # Reload best
        detector.load_state_dict(torch.load(ckpt, map_location=device))

        # Test evaluation — must use make_detector_input to build 6-channel tensors
        n_test = min(1000, len(x_test))
        x_test_adv = generate_adversarial_samples(art_victim, x_test, attack_type.lower(), args.eps, n_test)
        x_det = np.concatenate([make_detector_input(x_test[:n_test]),
                                 make_detector_input(x_test_adv)])
        y_det = np.concatenate([np.zeros(n_test, dtype=np.int64), np.ones(n_test, dtype=np.int64)])

        det_dataset = TensorDataset(torch.tensor(x_det), torch.tensor(y_det))
        det_loader  = DataLoader(det_dataset, batch_size=128, shuffle=False, num_workers=2)

        detector.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for imgs, lbls in det_loader:
                imgs = imgs.to(device)
                out = detector(imgs)
                probs_ = torch.softmax(out, 1)[:, 1].cpu().numpy()
                all_p.extend(probs_); all_l.extend(lbls.numpy())
        all_p = np.array(all_p); all_l = np.array(all_l)
        test_acc = ((all_p >= 0.5).astype(int) == all_l).mean()

        logger.info(f"[{attack_type}] Test Detection Accuracy: {test_acc*100:.2f}%")
        assert test_acc >= 0.70, f"{attack_type} detection acc {test_acc:.2%} < 70%!"

        curve_metrics = save_detector_plots(all_p, all_l, history, attack_type,
                                             args.save_dir, run)
        all_metrics[attack_type] = {
            'test_acc': test_acc,
            'best_val_acc': best_acc,
            **curve_metrics,
        }

        if run:
            run.log({f'{attack_type}/test_detection_acc': test_acc})

    # ── Final comparison plot ──
    if 'PGD' in all_metrics and 'BIM' in all_metrics:
        save_pgd_bim_comparison(all_metrics['PGD'], all_metrics['BIM'], args.save_dir, run)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'Attack':<10} {'Val Acc':>10} {'Test Det. Acc':>14} {'ROC AUC':>9} {'PR AUC':>8}")
    print('-' * 60)
    for atk, m in all_metrics.items():
        print(f"{atk:<10} {m['best_val_acc']*100:>9.2f}% {m['test_acc']*100:>13.2f}% "
              f"{m['roc_auc']*100:>8.2f}% {m['pr_auc']*100:>7.2f}%")

    if run: run.finish()


if __name__ == '__main__':
    main()