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

import timm
from huggingface_hub import login as hf_login

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
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)
    return train_loader, val_loader, val_set


# ─────────────────────────────── Model ───────────────────────────────────────

def build_baseline_vit(num_classes: int = 100) -> nn.Module:
    """ViT-S pre-trained on ImageNet, only classification head trainable."""
    # NOTE: hf_login moved to main() to avoid redundant API calls
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    
    # Freeze everything except the head
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[Baseline] Trainable: {trainable:,} / Total: {total:,}")
    return model


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.1):
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.scaling = alpha / rank
        in_f, out_f  = linear.in_features, linear.out_features

        self.lora_A   = nn.Linear(in_f,  rank,  bias=False)
        self.lora_B   = nn.Linear(rank,  out_f, bias=False)
        self.dropout  = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora_into_vit(model: nn.Module, rank: int, alpha: float,
                          dropout: float = 0.1) -> nn.Module:
    embed_dim = model.embed_dim 
    for block in model.blocks:
        attn = block.attn
        fused: nn.Linear = attn.qkv
        W = fused.weight.data
        b = fused.bias.data if fused.bias is not None else None
        D = embed_dim

        def make_frozen_linear(w_slice, b_slice):
            lin = nn.Linear(D, D, bias=(b_slice is not None))
            lin.weight = nn.Parameter(w_slice.clone(), requires_grad=False)
            if b_slice is not None:
                lin.bias = nn.Parameter(b_slice.clone(), requires_grad=False)
            return lin

        q_lin = make_frozen_linear(W[0*D:1*D], b[0*D:1*D] if b is not None else None)
        k_lin = make_frozen_linear(W[1*D:2*D], b[1*D:2*D] if b is not None else None)
        v_lin = make_frozen_linear(W[2*D:3*D], b[2*D:3*D] if b is not None else None)

        attn.lora_q = LoRALinear(q_lin, rank, alpha, dropout)
        attn.lora_k = LoRALinear(k_lin, rank, alpha, dropout)
        attn.lora_v = LoRALinear(v_lin, rank, alpha, dropout)

        def make_forward(a):
            def forward(x, attn_mask=None, is_causal=False, **kwargs):
                B, N, C = x.shape
                q, k, v = a.lora_q(x), a.lora_k(x), a.lora_v(x)
                h = a.num_heads
                scale = (C // h) ** -0.5
                q = q.reshape(B, N, h, C // h).transpose(1, 2)
                k = k.reshape(B, N, h, C // h).transpose(1, 2)
                v = v.reshape(B, N, h, C // h).transpose(1, 2)
                attn_w = (q @ k.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    attn_w = attn_w + attn_mask
                attn_w = attn_w.softmax(dim=-1)
                attn_w = a.attn_drop(attn_w)
                x_out  = (attn_w @ v).transpose(1, 2).reshape(B, N, C)
                x_out  = a.proj(x_out)
                x_out  = a.proj_drop(x_out)
                return x_out
            return forward
        attn.forward = make_forward(attn)
        if hasattr(attn, 'qkv'): del attn.qkv
    return model


def build_lora_vit(rank: int, alpha: float, dropout: float = 0.1,
                   num_classes: int = 100) -> nn.Module:
    """ViT-S/16 with LoRA. Login happens in main()."""
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    for p in model.parameters():
        p.requires_grad = False
    model = inject_lora_into_vit(model, rank, alpha, dropout)
    for p in model.head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"[LoRA r={rank} α={alpha}] Trainable: {trainable:,} / Total: {total:,} "
                f"({100*trainable/total:.3f}%)")
    return model


# ─────────────────────────────── Training ────────────────────────────────────

class GradientLogger:
    def __init__(self, model: nn.Module):
        self.history: Dict[str, list] = defaultdict(list)
        self._hooks = []
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                hook = param.register_hook(lambda grad, n=name: self.history[n].append(grad.norm().item()))
                self._hooks.append(hook)
    def remove(self):
        for h in self._hooks: h.remove()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(dim=1) == labels).sum().item()
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
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

def compute_classwise_accuracy(preds, labels, num_classes=100):
    acc = [ (preds[labels==c] == labels[labels==c]).mean() if (labels==c).sum()>0 else 0.0 for c in range(num_classes) ]
    return np.array(acc)


# ─────────────────────────────── Plots ───────────────────────────────────────

def save_loss_accuracy_curves(history: dict, save_dir: str, exp_name: str):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train'); axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train'); axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True)
    path = os.path.join(save_dir, f'{exp_name}_curves.png'); plt.savefig(path); plt.close(); return path

def save_classwise_histogram(classwise_acc, save_dir, exp_name, class_names):
    fig, ax = plt.subplots(figsize=(24, 6))
    ax.bar(range(100), classwise_acc * 100, color=plt.cm.RdYlGn(classwise_acc))
    ax.set_xticks(range(100)); ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    path = os.path.join(save_dir, f'{exp_name}_classwise.png'); plt.savefig(path); plt.close(); return path

def save_gradient_plot(grad_history, save_dir, exp_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    for n, v in list(grad_history.items())[:5]: ax.plot(v, label=n.split('.')[-1])
    ax.legend(); plt.savefig(os.path.join(save_dir, f'{exp_name}_grads.png')); plt.close()

def save_comparison_table_plot(results, save_dir):
    labels, accs = [r['name'] for r in results], [r['test_acc']*100 for r in results]
    plt.figure(figsize=(12, 6)); plt.bar(labels, accs); plt.xticks(rotation=45); plt.ylabel('Acc %')
    plt.savefig(os.path.join(save_dir, 'comparison.png')); plt.close()


# ─────────────────────────────── Main trainer ────────────────────────────────

def run_experiment(model_type, rank, alpha, dropout, epochs, lr, batch_size, save_dir, use_wandb=True, trial=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = f'baseline' if model_type == 'baseline' else f'lora_r{rank}_a{alpha}'
    out_dir = os.path.join(save_dir, exp_name); os.makedirs(out_dir, exist_ok=True)
    
    if use_wandb and trial is None:
        wandb.init(project='DLOps-Ass5-Q1', name=exp_name, config=locals())

    train_loader, val_loader, val_set = get_cifar100_loaders(batch_size)
    model = build_baseline_vit() if model_type == 'baseline' else build_lora_vit(rank, alpha, dropout)
    model = model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    grad_logger = GradientLogger(model) if model_type == 'lora' else None
    history = defaultdict(list)
    best_val_acc, best_ckpt = 0.0, os.path.join(out_dir, 'best.pt')

    for epoch in range(1, epochs + 1):
        tr_l, tr_a = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        va_l, va_a, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        for k, v in zip(['train_loss', 'val_loss', 'train_acc', 'val_acc'], [tr_l, va_l, tr_a, va_a]): history[k].append(v)
        
        if va_a > best_val_acc:
            best_val_acc = va_a
            torch.save(model.state_dict(), best_ckpt)

        if trial:
            trial.report(va_a, epoch)
            if trial.should_prune(): 
                if grad_logger: grad_logger.remove()
                raise optuna.TrialPruned()

    model.load_state_dict(torch.load(best_ckpt))
    _, test_a, test_p, test_l = evaluate(model, val_loader, criterion, device)
    
    save_loss_accuracy_curves(history, out_dir, exp_name)
    save_classwise_histogram(compute_classwise_accuracy(test_p, test_l), out_dir, exp_name, val_set.classes)
    if grad_logger:
        save_gradient_plot(grad_logger.history, out_dir, exp_name)
        grad_logger.remove()

    if use_wandb and trial is None: wandb.finish()
    
    return {'name': exp_name, 'test_acc': test_a, 'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


# ─────────────────────────────── Optuna ──────────────────────────────────────

def run_optuna(n_trials, epochs, save_dir):
    def objective(trial):
        r, a = trial.suggest_categorical('rank', [2,4,8]), trial.suggest_categorical('alpha', [2,4,8])
        dr, lr = trial.suggest_float('dropout', 0.0, 0.3), trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        res = run_experiment('lora', r, a, dr, epochs, lr, 128, save_dir, use_wandb=False, trial=trial)
        return res['test_acc']

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_trial.params


# ─────────────────────────────── Main ────────────────────────────────────────

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
    parser.add_argument('--skip_baseline', action='store_true')
    parser.add_argument('--skip_done', action='store_true')
    args = parser.parse_args()
    
    # CRITICAL: Login ONCE here to avoid connection resets during loops
    token = os.environ.get("HF_TOKEN", "")
    if token:
        logger.info("Authenticating with Hugging Face...")
        try:
            hf_login(token=token, add_to_git_credential=False)
        except Exception as e:
            logger.error(f"HF Login failed: {e}")

    os.makedirs(args.save_dir, exist_ok=True)
    results = []

    if args.mode in ('baseline', 'all'):
        results.append(run_experiment('baseline', None, None, 0.0, args.epochs, args.lr, args.batch_size, args.save_dir))

    if args.mode in ('lora', 'all'):
        for r in [2, 4, 8]:
            for a in [2, 4, 8]:
                results.append(run_experiment('lora', r, a, args.dropout, args.epochs, args.lr, args.batch_size, args.save_dir))

    if args.mode == 'optuna':
        bp = run_optuna(args.optuna_trials, min(args.epochs, 5), args.save_dir)
        run_experiment('lora', bp['rank'], bp['alpha'], bp.get('dropout', 0.1), args.epochs, bp.get('lr', args.lr), args.batch_size, args.save_dir)

    if len(results) > 1: save_comparison_table_plot(results, args.save_dir)

if __name__ == '__main__':
    main()
