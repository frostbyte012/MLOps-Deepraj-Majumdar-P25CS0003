import os
import numpy as np
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. CUSTOM DATASET  (Pillow instead of cv2)
# ─────────────────────────────────────────────
class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((128, 96), Image.NEAREST)
        img = np.array(img, dtype=np.float32) / 255.0

        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = mask.resize((128, 96), Image.NEAREST)
        mask = np.array(mask)
        mask = np.max(mask, axis=-1)

        img  = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        return img, mask


# ─────────────────────────────────────────────
# 2. UNET MODEL
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=23, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i // 2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1); x = self.ups[i + 1](x)
        return self.final(x)


# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
def compute_miou(preds, masks, num_classes=23):
    preds = preds.cpu().numpy(); masks = masks.cpu().numpy()
    iou_list = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (masks == cls)).sum()
        union = ((preds == cls) | (masks == cls)).sum()
        if union == 0: continue
        iou_list.append(inter / union)
    return float(np.mean(iou_list)) if iou_list else 0.0


def compute_mdice(preds, masks, num_classes=23):
    preds = preds.cpu().numpy(); masks = masks.cpu().numpy()
    dice_list = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (masks == cls)).sum()
        denom = (preds == cls).sum() + (masks == cls).sum()
        if denom == 0: continue
        dice_list.append(2 * inter / denom)
    return float(np.mean(dice_list)) if dice_list else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, num_classes=23):
    model.eval(); total_loss = total_iou = total_dice = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out   = model(imgs)
            preds = out.argmax(dim=1)
            total_loss += criterion(out, masks).item()
            total_iou  += compute_miou(preds, masks, num_classes)
            total_dice += compute_mdice(preds, masks, num_classes)
    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n


def main():
    IMAGE_DIR = "data/CameraRGB"
    MASK_DIR  = "data/CameraMask"
    OUT_DIR   = "Question2"
    os.makedirs(OUT_DIR, exist_ok=True)

    exts = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = sorted([p for e in exts for p in glob.glob(os.path.join(IMAGE_DIR, "**", e), recursive=True)])
    mask_paths  = sorted([p for e in exts for p in glob.glob(os.path.join(MASK_DIR,  "**", e), recursive=True)])

    assert len(image_paths) == len(mask_paths) and len(image_paths) > 0, \
        "Found %d images and %d masks" % (len(image_paths), len(mask_paths))
    print("Total samples: %d" % len(image_paths))

    tr_imgs, te_imgs, tr_masks, te_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42)

    train_loader = DataLoader(CityscapesDataset(tr_imgs, tr_masks), batch_size=16, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(CityscapesDataset(te_imgs, te_masks), batch_size=16, shuffle=False, num_workers=2)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    model     = UNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    EPOCHS  = 15
    history = {"loss": [], "miou": [], "mdice": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, iou, dice = evaluate(model, train_loader, criterion, device)
        scheduler.step()
        history["loss"].append(tr_loss)
        history["miou"].append(iou)
        history["mdice"].append(dice)
        print("Epoch %02d/%d | Loss: %.4f | mIOU: %.4f | mDice: %.4f" % (epoch, EPOCHS, tr_loss, iou, dice))

    _, test_iou, test_dice = evaluate(model, test_loader, criterion, device)
    print("\nTest mIOU : %.4f" % test_iou)
    print("Test mDice: %.4f" % test_dice)

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "unet_cityscapes.pth"))
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"test_miou": test_iou, "test_mdice": test_dice,
                   "history": history,
                   "test_image_paths": te_imgs,
                   "test_mask_paths":  te_masks}, f, indent=2)

    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_range, history["loss"],  "b-o", markersize=4); axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs_range, history["miou"],  "g-o", markersize=4); axes[1].set_title("mIOU"); axes[1].set_xlabel("Epoch")
    axes[1].axhline(test_iou,  color="r", linestyle="--", label="Test %.4f" % test_iou); axes[1].legend()
    axes[2].plot(epochs_range, history["mdice"], "m-o", markersize=4); axes[2].set_title("mDice"); axes[2].set_xlabel("Epoch")
    axes[2].axhline(test_dice, color="r", linestyle="--", label="Test %.4f" % test_dice); axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_plots.png"), dpi=150)
    plt.close()
    print("Done! Plots saved to %s/training_plots.png" % OUT_DIR)


if __name__ == "__main__":
    main()