"""
Assignment 5 - Q1: ViT-S Fine-tuning with LoRA on CIFAR-100
Supports: baseline (no LoRA), LoRA with various ranks/alphas, Optuna HPO
"""

import os
import argparse
import json
import math
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from collections import defaultdict

import wandb
import optuna
from optuna.trial import Trial

from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
import timm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────── Data ────────────────────────────────────────

def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 4):
    """CIFAR-100 with strong augmentation for training."""
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(224),           # ViT expects 224×224
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=train_transform)
    val_set   = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, val_set


# ─────────────────────────────── Model ───────────────────────────────────────

def build_baseline_vit(num_classes: int = 100) -> nn.Module:
    """ViT-S pre-trained on ImageNet, only classification head trainable."""
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    # Freeze everything except the head
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[Baseline] Trainable: {trainable:,} / Total: {total:,}")
    return model


def build_lora_vit(rank: int, alpha: float, dropout: float = 0.1,
                   num_classes: int = 100) -> nn.Module:
    """ViT-S with LoRA injected into Q, K, V attention weights + trainable head."""
    # Load base model via timm, then wrap with PEFT
    # We use HuggingFace ViT for PEFT compatibility
    config = ViTConfig.from_pretrained('WinKawaks/vit-small-patch16-224')
    config.num_labels = num_classes
    model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-small-patch16-224',
        config=config,
        ignore_mismatched_sizes=True,
    )

    # LoRA configuration targeting Q, K, V projections
    target_modules = ["query", "key", "value"]
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # SEQ_CLS passes input_ids; FEATURE_EXTRACTION passes pixel_values correctly for ViT
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["classifier"],  # keep classifier trainable
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[LoRA r={rank} α={alpha}] Trainable params: {trainable:,}")
    return model


# ─────────────────────────────── Training ────────────────────────────────────

class GradientLogger:
    """Tracks gradient norms on LoRA weight matrices during training."""
    def __init__(self, model: nn.Module):
        self.history: Dict[str, list] = defaultdict(list)
        self._hooks = []
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self.history[n].append(grad.norm().item())
                )
                self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast('cuda'):   # torch.cuda.amp.autocast deprecated ≥ 2.1
                out = model(imgs)
                logits = out.logits if hasattr(out, 'logits') else out
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        logits = out.logits if hasattr(out, 'logits') else out
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def compute_classwise_accuracy(preds, labels, num_classes=100):
    acc = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            acc.append(0.0)
        else:
            acc.append((preds[mask] == labels[mask]).mean())
    return np.array(acc)


# ─────────────────────────────── Plots ───────────────────────────────────────

def save_loss_accuracy_curves(history: dict, save_dir: str, exp_name: str):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val Loss')
    axes[0].set_title(f'{exp_name} — Loss Curve')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc')
    axes[1].set_title(f'{exp_name} — Accuracy Curve')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{exp_name}_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_classwise_histogram(classwise_acc: np.ndarray, save_dir: str,
                              exp_name: str, class_names: list):
    fig, ax = plt.subplots(figsize=(24, 6))
    colors = plt.cm.RdYlGn(classwise_acc)
    bars = ax.bar(range(100), classwise_acc * 100, color=colors)
    ax.set_xticks(range(100))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{exp_name} — Class-wise Test Accuracy')
    ax.axhline(y=classwise_acc.mean() * 100, color='blue', linestyle='--', label=f'Mean: {classwise_acc.mean()*100:.1f}%')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f'{exp_name}_classwise.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_gradient_plot(grad_history: dict, save_dir: str, exp_name: str):
    if not grad_history:
        return None
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, vals in list(grad_history.items())[:8]:  # top 8 layers
        short = name.split('.')[-3] + '.' + name.split('.')[-1]
        ax.plot(vals, label=short, alpha=0.7)
    ax.set_title(f'{exp_name} — Gradient Norms on LoRA Weights')
    ax.set_xlabel('Step'); ax.set_ylabel('Gradient Norm')
    ax.legend(fontsize=7); ax.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{exp_name}_gradients.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_comparison_table_plot(results: list, save_dir: str):
    """Bar chart comparing overall test accuracy across all experiments."""
    labels  = [r['name'] for r in results]
    accs    = [r['test_acc'] * 100 for r in results]
    params  = [r['trainable_params'] for r in results]

    fig, ax1 = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))
    bars = ax1.bar(x, accs, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)', color='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, params, 'r-o', label='Trainable Params')
    ax2.set_ylabel('Trainable Parameters', color='red')

    ax1.set_title('Comparison: Test Accuracy & Trainable Parameters Across Experiments')
    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison_bar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ─────────────────────────────── Main trainer ────────────────────────────────

def run_experiment(
    model_type: str,
    rank: Optional[int],
    alpha: Optional[float],
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    save_dir: str,
    use_wandb: bool = True,
    wandb_project: str = 'DLOps-Ass5-Q1',
    trial: Optional[Trial] = None,
) -> dict:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = f'baseline' if model_type == 'baseline' else f'lora_r{rank}_a{alpha}'
    save_dir = os.path.join(save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"\n{'='*60}\nStarting: {exp_name}\n{'='*60}")

    if use_wandb and trial is None:
        wandb.init(project=wandb_project, name=exp_name, config={
            'model_type': model_type, 'rank': rank, 'alpha': alpha,
            'dropout': dropout, 'epochs': epochs, 'lr': lr,
        })

    train_loader, val_loader, val_set = get_cifar100_loaders(batch_size)
    class_names = val_set.classes

    if model_type == 'baseline':
        model = build_baseline_vit()
    else:
        model = build_lora_vit(rank, alpha, dropout)

    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None  # torch.cuda.amp deprecated ≥ 2.1

    # Gradient logger for LoRA experiments
    grad_logger = None
    if model_type == 'lora':
        grad_logger = GradientLogger(model)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_ckpt = os.path.join(save_dir, f'{exp_name}_best.pt')

    print(f"\n{'Epoch':>5} {'TrLoss':>8} {'VaLoss':>8} {'TrAcc':>8} {'VaAcc':>8}")
    print('-' * 45)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>5} {tr_loss:>8.4f} {va_loss:>8.4f} {tr_acc*100:>7.2f}% {va_acc*100:>7.2f}%  ({elapsed:.1f}s)")

        if use_wandb and trial is None:
            wandb.log({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss,
                       'train_acc': tr_acc, 'val_acc': va_acc, 'lr': scheduler.get_last_lr()[0]})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_ckpt)

        # Optuna pruning
        if trial is not None:
            trial.report(va_acc, epoch)
            if trial.should_prune():
                if grad_logger: grad_logger.remove()
                raise optuna.TrialPruned()

    # ─── Final test evaluation ───
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    _, test_acc, test_preds, test_labels = evaluate(model, val_loader, criterion, device)
    classwise_acc = compute_classwise_accuracy(test_preds, test_labels)

    # ─── Plots ───
    curve_path = save_loss_accuracy_curves(history, save_dir, exp_name)
    hist_path  = save_classwise_histogram(classwise_acc, save_dir, exp_name, class_names)
    grad_path  = None
    if grad_logger:
        grad_path = save_gradient_plot(grad_logger.history, save_dir, exp_name)
        grad_logger.remove()

    if use_wandb and trial is None:
        wandb.log({
            'test_accuracy': test_acc,
            'classwise_accuracy_histogram': wandb.Image(hist_path),
            'loss_accuracy_curves': wandb.Image(curve_path),
        })
        if grad_path:
            wandb.log({'gradient_norms': wandb.Image(grad_path)})
        wandb.finish()

    # ─── Save results ───
    result = {
        'name': exp_name,
        'model_type': model_type,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'trainable_params': trainable_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'history': history,
        'classwise_acc': classwise_acc.tolist(),
    }
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump({k: v for k, v in result.items() if k != 'classwise_acc'}, f, indent=2)

    logger.info(f"[{exp_name}] Best Val: {best_val_acc*100:.2f}% | Test: {test_acc*100:.2f}%")
    return result


# ─────────────────────────────── Optuna HPO ──────────────────────────────────

def run_optuna(n_trials: int = 20, epochs: int = 5, save_dir: str = './results'):
    def objective(trial: Trial) -> float:
        rank    = trial.suggest_categorical('rank',    [2, 4, 8])
        alpha   = trial.suggest_categorical('alpha',   [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        lr      = trial.suggest_float('lr', 1e-5, 1e-3, log=True)

        result = run_experiment(
            model_type='lora', rank=rank, alpha=alpha, dropout=dropout,
            epochs=epochs, lr=lr, batch_size=128, save_dir=save_dir,
            use_wandb=False, trial=trial,
        )
        return result['best_val_acc']

    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"\nBest trial: {study.best_trial.params}")
    logger.info(f"Best Val Acc: {study.best_value*100:.2f}%")

    # Plot Optuna results
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_contour,
        )
        fig = plot_optimization_history(study)
        plt.savefig(os.path.join(save_dir, 'optuna_history.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig = plot_param_importances(study)
        plt.savefig(os.path.join(save_dir, 'optuna_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Optuna plot error: {e}")

    return study.best_trial.params


# ─────────────────────────────── CLI ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ViT-S LoRA CIFAR-100 Training')
    parser.add_argument('--mode',       choices=['baseline', 'lora', 'all', 'optuna'], default='all')
    parser.add_argument('--rank',       type=int,   default=4)
    parser.add_argument('--alpha',      type=float, default=4)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--save_dir',   type=str,   default='./results/Q1')
    parser.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q1')
    parser.add_argument('--no_wandb',   action='store_true')
    parser.add_argument('--optuna_trials', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    use_wandb = not args.no_wandb

    all_results = []

    if args.mode in ('baseline', 'all'):
        res = run_experiment('baseline', None, None, 0.0, args.epochs, args.lr,
                             args.batch_size, args.save_dir, use_wandb, args.wandb_project)
        all_results.append(res)

    if args.mode in ('lora', 'all'):
        ranks  = [2, 4, 8]
        alphas = [2, 4, 8]
        for r in ranks:
            for a in alphas:
                res = run_experiment('lora', r, a, args.dropout, args.epochs, args.lr,
                                     args.batch_size, args.save_dir, use_wandb, args.wandb_project)
                all_results.append(res)

    if args.mode == 'optuna':
        best_params = run_optuna(args.optuna_trials, min(args.epochs, 5), args.save_dir)
        # Run best config for full epochs
        run_experiment('lora', best_params['rank'], best_params['alpha'],
                       best_params.get('dropout', 0.1), args.epochs,
                       best_params.get('lr', 1e-4), args.batch_size, args.save_dir,
                       use_wandb, args.wandb_project + '-best')

    if len(all_results) > 1:
        bar_path = save_comparison_table_plot(all_results, args.save_dir)
        # Print summary table
        print(f"\n{'='*90}")
        print(f"{'Experiment':<25} {'Rank':>5} {'Alpha':>6} {'Dropout':>8} {'Test Acc':>10} {'Trainable Params':>17}")
        print('-' * 90)
        for r in all_results:
            print(f"{r['name']:<25} {str(r['rank']):>5} {str(r['alpha']):>6} "
                  f"{str(r['dropout']):>8} {r['test_acc']*100:>9.2f}% {r['trainable_params']:>17,}")
        print('=' * 90)


if __name__ == '__main__':
    main()