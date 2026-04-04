"""
Assignment 5 - Q2(i): FGSM Attack — From Scratch vs IBM ART
ResNet18 trained on CIFAR-10, then attacked with FGSM both ways.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
import logging

# IBM ART imports
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────── Data ───────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────── Model ──────────────────────────────────────────

def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)          # pretrained=False is deprecated in torchvision ≥ 0.13
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ──────────────────────────── Training ───────────────────────────────────────

def train_model(model, train_loader, test_loader, epochs, lr, save_path, device,
                wandb_run=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None  # torch.cuda.amp deprecated ≥ 2.1

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):   # torch.cuda.amp.autocast deprecated ≥ 2.1
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
            tr_loss += loss.item() * labels.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate_clean(model, test_loader, criterion, device)
        logger.info(f"Epoch {epoch:>3} | TrLoss:{tr_loss/tr_total:.4f} | ValLoss:{val_loss:.4f} | ValAcc:{val_acc*100:.2f}%")

        if wandb_run:
            wandb_run.log({'epoch': epoch, 'train_loss': tr_loss/tr_total,
                           'val_loss': val_loss, 'val_acc': val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    logger.info(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    return best_acc


@torch.no_grad()
def evaluate_clean(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


# ──────────────────────────── FGSM from Scratch ──────────────────────────────

def fgsm_attack_scratch(model, imgs, labels, epsilon, device):
    """Pure PyTorch FGSM implementation."""
    imgs_adv = imgs.clone().detach().requires_grad_(True).to(device)
    criterion = nn.CrossEntropyLoss()
    out = model(imgs_adv)
    loss = criterion(out, labels.to(device))
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * imgs_adv.grad.sign()
    imgs_adv = imgs_adv + perturbation
    # Clip to valid range (assuming normalized)
    imgs_adv = torch.clamp(imgs_adv.detach(), -3, 3)
    return imgs_adv


def evaluate_fgsm_scratch(model, loader, epsilon, device, max_batches=None):
    """Evaluate model accuracy under FGSM (from scratch).
    NOTE: must NOT use @torch.no_grad() here — fgsm_attack_scratch needs
    gradients to compute the sign of the loss w.r.t. the input.
    We disable grad only for the final inference step after the attack.
    """
    model.eval()
    correct, total = 0, 0
    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs_adv = fgsm_attack_scratch(model, imgs, labels, epsilon, device)
        with torch.no_grad():   # safe here — attack is already done
            out = model(imgs_adv)
        correct += (out.argmax(1) == labels.to(device)).sum().item()
        total   += labels.size(0)
    return correct / total


# ──────────────────────────── FGSM via IBM ART ───────────────────────────────

def get_art_classifier(model, device):
    """Wrap PyTorch model in ART PyTorchClassifier."""
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1).astype(np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1).astype(np.float32)

    criterion = nn.CrossEntropyLoss()
    optimizer_art = optim.SGD(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer_art,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(-3.0, 3.0),
        device_type='gpu' if torch.cuda.is_available() else 'cpu',
    )
    return classifier


def evaluate_fgsm_art(classifier, loader, epsilon, max_batches=None):
    """Evaluate model accuracy under FGSM using IBM ART."""
    attack = FastGradientMethod(estimator=classifier, eps=epsilon,
                                 eps_step=epsilon, targeted=False, batch_size=128)
    correct, total = 0, 0
    all_imgs, all_adv_imgs, all_labels = [], [], []

    for i, (imgs, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x_np = imgs.numpy()
        y_np = labels.numpy()
        x_adv = attack.generate(x=x_np)
        preds = np.argmax(classifier.predict(x_adv), axis=1)
        correct += (preds == y_np).sum()
        total   += len(y_np)
        if i == 0:  # Save first batch for visualization
            all_imgs.append(x_np[:10])
            all_adv_imgs.append(x_adv[:10])
            all_labels.append(y_np[:10])

    return correct / total, (all_imgs[0] if all_imgs else None,
                              all_adv_imgs[0] if all_adv_imgs else None,
                              all_labels[0] if all_labels else None)


# ──────────────────────────── Visualization ──────────────────────────────────

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def denormalize(img_tensor, mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)):
    """Convert normalized tensor/array to displayable image."""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor.copy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 1)
    return img.transpose(1, 2, 0)


def save_fgsm_comparison(imgs_clean, imgs_adv_scratch, imgs_adv_art,
                          labels, model, epsilon, save_dir, device):
    """Save visual comparison: Original | Scratch-FGSM | ART-FGSM."""
    model.eval()
    n = min(10, len(labels))
    fig, axes = plt.subplots(n, 4, figsize=(16, 2.5 * n))

    with torch.no_grad():
        clean_tensor   = torch.tensor(imgs_clean).to(device)
        scratch_tensor = fgsm_attack_scratch(model,
                                              torch.tensor(imgs_clean),
                                              torch.tensor(labels), epsilon, device)

    for i in range(n):
        label_name = CIFAR10_CLASSES[labels[i]]

        # Original
        axes[i][0].imshow(denormalize(imgs_clean[i]))
        axes[i][0].set_title(f'Clean\nTrue: {label_name}', fontsize=8)
        axes[i][0].axis('off')

        # Scratch FGSM
        scratch_img = scratch_tensor[i].cpu().numpy()
        axes[i][1].imshow(denormalize(scratch_img))
        with torch.no_grad():
            pred = model(scratch_tensor[i:i+1]).argmax().item()
        axes[i][1].set_title(f'FGSM Scratch\nPred: {CIFAR10_CLASSES[pred]}', fontsize=8)
        axes[i][1].axis('off')

        # ART FGSM
        axes[i][2].imshow(denormalize(imgs_adv_art[i]))
        with torch.no_grad():
            art_t = torch.tensor(imgs_adv_art[i:i+1]).to(device)
            pred_art = model(art_t).argmax().item()
        axes[i][2].set_title(f'FGSM ART\nPred: {CIFAR10_CLASSES[pred_art]}', fontsize=8)
        axes[i][2].axis('off')

        # Perturbation (amplified)
        pert = np.abs(imgs_adv_art[i] - imgs_clean[i])
        pert = (pert / pert.max()) if pert.max() > 0 else pert
        axes[i][3].imshow(pert.transpose(1, 2, 0))
        axes[i][3].set_title('Perturbation\n(amplified)', fontsize=8)
        axes[i][3].axis('off')

    plt.suptitle(f'FGSM Comparison (ε={epsilon})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'fgsm_comparison_eps{epsilon:.3f}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_epsilon_sweep_plot(results, save_dir):
    """Plot accuracy vs perturbation strength for all attack types."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epsilons = results['epsilons']
    ax.plot(epsilons, [a * 100 for a in results['clean']],   'g--',  label='Clean (baseline)', linewidth=2)
    ax.plot(epsilons, [a * 100 for a in results['scratch']], 'b-o',  label='FGSM Scratch', linewidth=2)
    ax.plot(epsilons, [a * 100 for a in results['art']],     'r-^',  label='FGSM ART', linewidth=2)
    ax.set_xlabel('Perturbation Strength (ε)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Perturbation Strength vs Accuracy Drop', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True)
    path = os.path.join(save_dir, 'epsilon_sweep.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ──────────────────────────── Main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',    type=int,   default=50)
    parser.add_argument('--lr',        type=float, default=0.1)
    parser.add_argument('--batch_size',type=int,   default=128)
    parser.add_argument('--save_dir',  type=str,   default='./results/Q2i')
    parser.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    parser.add_argument('--no_wandb',  action='store_true')
    parser.add_argument('--skip_train',action='store_true', help='Skip training if weights exist')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    run = None
    if not args.no_wandb:
        run = wandb.init(project=args.wandb_project, name='fgsm-comparison')

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    ckpt_path = os.path.join(args.save_dir, 'resnet18_clean.pt')

    # ── Step 1: Train clean model ──
    model = build_resnet18().to(device)
    if args.skip_train and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info("Loaded existing weights.")
    else:
        logger.info("Training ResNet18 on CIFAR-10...")
        clean_acc = train_model(model, train_loader, test_loader, args.epochs,
                                args.lr, ckpt_path, device, run)
        assert clean_acc >= 0.72, f"Clean accuracy {clean_acc:.2%} < 72%. Try more epochs."

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    _, clean_acc = evaluate_clean(model, test_loader, criterion, device)
    logger.info(f"Clean Test Accuracy: {clean_acc*100:.2f}%")

    # ── Step 2 & 3: FGSM sweep ──
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
    art_cls  = get_art_classifier(model, device)

    sweep = {'epsilons': epsilons, 'clean': [], 'scratch': [], 'art': []}
    vis_data = None

    for eps in epsilons:
        logger.info(f"\n── ε = {eps:.3f} ──")
        if eps == 0:
            sweep['clean'].append(clean_acc)
            sweep['scratch'].append(clean_acc)
            sweep['art'].append(clean_acc)
        else:
            # Scratch
            sc_acc = evaluate_fgsm_scratch(model, test_loader, eps, device, max_batches=40)
            # ART
            art_acc, vis = evaluate_fgsm_art(art_cls, test_loader, eps, max_batches=40)
            sweep['clean'].append(clean_acc)
            sweep['scratch'].append(sc_acc)
            sweep['art'].append(art_acc)
            logger.info(f"Scratch: {sc_acc*100:.2f}% | ART: {art_acc*100:.2f}%")
            if eps == 0.03 and vis[0] is not None:
                vis_data = vis

    # ── Step 4: Visual comparison ──
    if vis_data is not None and vis_data[0] is not None:
        vis_path = save_fgsm_comparison(
            vis_data[0], None, vis_data[1], vis_data[2],
            model, 0.03, args.save_dir, device
        )
        if run: run.log({'fgsm_comparison': wandb.Image(vis_path)})

    # ── Step 5: Epsilon sweep plot ──
    sweep_path = save_epsilon_sweep_plot(sweep, args.save_dir)
    if run: run.log({'epsilon_sweep': wandb.Image(sweep_path)})

    # ── Log to WandB: 10 sample images ──
    if run:
        imgs_iter = iter(test_loader)
        imgs_batch, labels_batch = next(imgs_iter)
        imgs_batch = imgs_batch[:10]
        labels_batch = labels_batch[:10]
        imgs_adv_art_sample, _, _ = evaluate_fgsm_art(
            art_cls, [(imgs_batch.numpy(), labels_batch.numpy())], 0.03)
        for i in range(10):
            clean_img = wandb.Image(denormalize(imgs_batch[i].numpy()),
                                    caption=f"Clean: {CIFAR10_CLASSES[labels_batch[i]]}")
            run.log({f'fgsm_clean_sample_{i}': clean_img})

    # ── Print summary ──
    eps_report = [0.0, 0.01, 0.03, 0.05, 0.1]
    print(f"\n{'ε':>6} {'Clean Acc':>10} {'FGSM Scratch':>14} {'FGSM ART':>10} {'Drop (Scratch)':>15} {'Drop (ART)':>11}")
    print('-' * 70)
    for i, eps in enumerate(sweep['epsilons']):
        if eps in eps_report:
            drop_sc  = (sweep['clean'][i] - sweep['scratch'][i]) * 100
            drop_art = (sweep['clean'][i] - sweep['art'][i]) * 100
            print(f"{eps:>6.3f} {sweep['clean'][i]*100:>9.2f}% {sweep['scratch'][i]*100:>13.2f}% "
                  f"{sweep['art'][i]*100:>9.2f}% {drop_sc:>13.2f}% {drop_art:>10.2f}%")

    if run: run.finish()


if __name__ == '__main__':
    main()