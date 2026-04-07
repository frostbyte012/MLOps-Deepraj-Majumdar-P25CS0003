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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report
import wandb
import logging

from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def get_cifar10_numpy(train: bool = True):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    ds = torchvision.datasets.CIFAR10('./data', train=train, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in loader: xs.append(x.numpy()); ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)

def build_victim_resnet18(num_classes=10):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_art_victim(model, device):
    return PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(), optimizer=optim.SGD(model.parameters(), lr=0.01), input_shape=(3, 32, 32), nb_classes=10, clip_values=(-3.0, 3.0), device_type='gpu' if torch.cuda.is_available() else 'cpu')

def generate_adversarial_samples(art_classifier, x_clean: np.ndarray, attack_type: str, eps: float, n_samples: int) -> np.ndarray:
    x = x_clean[:n_samples]
    if attack_type == 'pgd': attack = ProjectedGradientDescent(estimator=art_classifier, eps=eps, eps_step=eps/10, max_iter=40, targeted=False, batch_size=256)
    else: attack = BasicIterativeMethod(estimator=art_classifier, eps=eps, eps_step=eps/10, max_iter=40, targeted=False, batch_size=256)
    logger.info(f"Generating {attack_type.upper()} adversarial samples (n={n_samples})...")
    return attack.generate(x=x)

def build_detector_resnet34() -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def make_detector_input(x: np.ndarray) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    blurred  = uniform_filter(x, size=(1, 1, 3, 3))
    residual = np.abs(x - blurred) * 10.0
    return residual.astype(np.float32)

def build_detector_dataset(x_clean, x_adv):
    n = min(len(x_clean), len(x_adv))
    split = int(0.8 * n)
    x_c_tr = make_detector_input(x_clean[:split])
    x_a_tr = make_detector_input(x_adv[:split])
    x_c_va = make_detector_input(x_clean[split:n])
    x_a_va = make_detector_input(x_adv[split:n])
    
    x_tr = np.concatenate([x_c_tr, x_a_tr], axis=0)
    y_tr = np.concatenate([np.zeros(split, dtype=np.int64), np.ones(split, dtype=np.int64)])
    x_va = np.concatenate([x_c_va, x_a_va], axis=0)
    y_va = np.concatenate([np.zeros(n-split, dtype=np.int64), np.ones(n-split, dtype=np.int64)])
    
    perm_tr = np.random.permutation(len(x_tr))
    x_tr, y_tr = x_tr[perm_tr], y_tr[perm_tr]
    perm_va = np.random.permutation(len(x_va))
    x_va, y_va = x_va[perm_va], y_va[perm_va]
    
    return (x_tr, y_tr), (x_va, y_va)

def train_detector(model, train_data, val_data, epochs, lr, save_path, device, attack_type, wandb_run=None):
    x_tr, y_tr = train_data; x_va, y_va = val_data
    tr_loader = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)), batch_size=128, shuffle=True, num_workers=0)
    va_loader = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)), batch_size=128, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0; history = {'tr_loss': [], 'va_loss': [], 'tr_acc': [], 'va_acc': []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(); out = model(imgs); loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            tr_loss += loss.item() * labels.size(0); tr_correct += (out.argmax(1) == labels).sum().item(); tr_total += labels.size(0)

        model.eval()
        va_loss, va_correct, va_total = 0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in va_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs); loss = criterion(out, labels)
                va_loss += loss.item() * labels.size(0); va_correct += (out.argmax(1) == labels).sum().item(); va_total += labels.size(0)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

        tr_acc = tr_correct / tr_total; va_acc = va_correct / va_total
        history['tr_loss'].append(tr_loss / tr_total); history['va_loss'].append(va_loss / va_total)
        history['tr_acc'].append(tr_acc); history['va_acc'].append(va_acc)
        logger.info(f"[{attack_type}] Epoch {epoch:>3} | TrAcc:{tr_acc*100:.2f}% | VaAcc:{va_acc*100:.2f}%")

        if va_acc > best_acc: best_acc = va_acc; torch.save(model.state_dict(), save_path)
    return best_acc, history, np.array(all_probs), np.array(all_labels)

def save_detector_plots(all_probs, all_labels, history, attack_type, save_dir, wandb_run=None):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['tr_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history['tr_loss'], 'b-o', label='Train'); axes[0].plot(epochs, history['va_loss'], 'r-o', label='Val')
    axes[0].set_title(f'{attack_type} Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, [a*100 for a in history['tr_acc']], 'b-o', label='Train'); axes[1].plot(epochs, [a*100 for a in history['va_acc']], 'r-o', label='Val')
    axes[1].set_title(f'{attack_type} Accuracy'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); curves_path = os.path.join(save_dir, f'{attack_type}_curves.png'); plt.savefig(curves_path, dpi=150); plt.close()

    preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Adv'], yticklabels=['Clean', 'Adv'], ax=ax)
    ax.set_title(f'{attack_type} CM'); plt.tight_layout(); cm_path = os.path.join(save_dir, f'{attack_type}_confusion.png'); plt.savefig(cm_path, dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(all_labels, all_probs); roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.3f}'); ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(f'{attack_type} ROC'); ax.legend(); ax.grid(True); plt.tight_layout(); roc_path = os.path.join(save_dir, f'{attack_type}_roc.png'); plt.savefig(roc_path, dpi=150); plt.close()

    prec, rec, _ = precision_recall_curve(all_labels, all_probs); pr_auc = auc(rec, prec)
    if wandb_run: wandb_run.log({f'{attack_type}/curves': wandb.Image(curves_path), f'{attack_type}/roc_auc': roc_auc, f'{attack_type}/pr_auc': pr_auc})
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

def save_pgd_bim_comparison(pgd_metrics, bim_metrics, save_dir, wandb_run=None):
    metrics = ['Detection Acc', 'ROC AUC', 'PR AUC']
    pgd_vals = [pgd_metrics['test_acc']*100, pgd_metrics['roc_auc']*100, pgd_metrics['pr_auc']*100]
    bim_vals = [bim_metrics['test_acc']*100, bim_metrics['roc_auc']*100, bim_metrics['pr_auc']*100]
    x = np.arange(len(metrics)); width = 0.35; fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width/2, pgd_vals, width, label='PGD'); ax.bar(x + width/2, bim_vals, width, label='BIM')
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12); ax.set_title('Adversarial Detector Comparison', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(axis='y'); plt.tight_layout(); path = os.path.join(save_dir, 'pgd_vs_bim_comparison.png'); plt.savefig(path, dpi=150); plt.close()
    if wandb_run: wandb_run.log({'pgd_vs_bim_comparison': wandb.Image(path)})
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--victim_ckpt', type=str, default='./results/Q2i/resnet18_clean.pt')
    parser.add_argument('--epochs',     type=int,   default=15)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--n_samples',  type=int,   default=5000)
    parser.add_argument('--eps',        type=float, default=0.03)
    parser.add_argument('--save_dir',   type=str,   default='./results/Q2ii')
    parser.add_argument('--wandb_project', type=str, default='DLOps-Ass5-Q2')
    parser.add_argument('--no_wandb',   action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = wandb.init(project=args.wandb_project, name='adversarial-detection') if not args.no_wandb else None

    victim = build_victim_resnet18().to(device); victim.load_state_dict(torch.load(args.victim_ckpt, map_location=device)); victim.eval()
    art_victim = get_art_victim(victim, device)

    x_train, _ = get_cifar10_numpy(train=True)
    x_test,  _ = get_cifar10_numpy(train=False)

    pgd_cache = os.path.join(args.save_dir, f'pgd_adv_eps{args.eps:.3f}_v5.npy')
    bim_cache = os.path.join(args.save_dir, f'bim_adv_eps{args.eps:.3f}_v5.npy')

    if os.path.exists(pgd_cache): x_train_pgd = np.load(pgd_cache)
    else: x_train_pgd = generate_adversarial_samples(art_victim, x_train, 'pgd', args.eps, args.n_samples); np.save(pgd_cache, x_train_pgd)
    if os.path.exists(bim_cache): x_train_bim = np.load(bim_cache)
    else: x_train_bim = generate_adversarial_samples(art_victim, x_train, 'bim', args.eps, args.n_samples); np.save(bim_cache, x_train_bim)

    all_metrics = {}
    for attack_type, x_adv in [('PGD', x_train_pgd), ('BIM', x_train_bim)]:
        logger.info(f"\n{'='*60}\nTraining {attack_type} Detector\n{'='*60}")
        tr_data, va_data = build_detector_dataset(x_train[:args.n_samples], x_adv)
        detector = build_detector_resnet34().to(device)
        ckpt = os.path.join(args.save_dir, f'detector_{attack_type.lower()}.pt')
        best_acc, history, probs, labels = train_detector(detector, tr_data, va_data, args.epochs, args.lr, ckpt, device, attack_type, run)
        detector.load_state_dict(torch.load(ckpt, map_location=device))

        n_test = min(1000, len(x_test))
        x_test_adv = generate_adversarial_samples(art_victim, x_test, attack_type.lower(), args.eps, n_test)
        x_det = np.concatenate([make_detector_input(x_test[:n_test]), make_detector_input(x_test_adv)])
        y_det = np.concatenate([np.zeros(n_test, dtype=np.int64), np.ones(n_test, dtype=np.int64)])
        det_loader = DataLoader(TensorDataset(torch.tensor(x_det), torch.tensor(y_det)), batch_size=128, shuffle=False, num_workers=0)
        
        detector.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for imgs, lbls in det_loader:
                out = detector(imgs.to(device))
                all_p.extend(torch.softmax(out, 1)[:, 1].cpu().numpy()); all_l.extend(lbls.numpy())
        
        test_acc = ((np.array(all_p) >= 0.5).astype(int) == np.array(all_l)).mean()
        logger.info(f"[{attack_type}] Test Detection Accuracy: {test_acc*100:.2f}%")
        assert test_acc >= 0.65, f"{attack_type} detection acc {test_acc:.2%} < 65%!"
        
        curve_metrics = save_detector_plots(np.array(all_p), np.array(all_l), history, attack_type, args.save_dir, run)
        all_metrics[attack_type] = {'test_acc': test_acc, 'best_val_acc': best_acc, **curve_metrics}

    if 'PGD' in all_metrics and 'BIM' in all_metrics: save_pgd_bim_comparison(all_metrics['PGD'], all_metrics['BIM'], args.save_dir, run)
    if run: run.finish()

if __name__ == '__main__':
    main()
