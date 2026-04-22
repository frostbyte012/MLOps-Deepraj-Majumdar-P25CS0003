import os
import json
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# ─── UNet (identical to train.py) ───────────────
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
    def __init__(self, in_channels=3, num_classes=23, features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1); x = self.ups[i+1](x)
        return self.final(x)

# ─── Constants ──────────────────────────────────
NUM_CLASSES  = 23
MODEL_PATH   = "Question2/unet_cityscapes.pth"
RESULTS_PATH = "Question2/results.json"
PLOTS_PATH   = "Question2/training_plots.png"

PALETTE = np.array([
    list(mcolors.to_rgb(c))
    for c in (list(mcolors.TABLEAU_COLORS.values()) +
              list(mcolors.CSS4_COLORS.values()))
], dtype=np.float32)[:NUM_CLASSES]

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model  = UNet(num_classes=NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

def preprocess(pil_img):
    img = pil_img.resize((128, 96), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def mask_to_color(mask_2d):
    h, w = mask_2d.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color[mask_2d == cls] = (PALETTE[cls] * 255).astype(np.uint8)
    return color

def predict(model, device, pil_img):
    tensor = preprocess(pil_img).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.argmax(dim=1).squeeze(0).cpu().numpy()

# ─── APP ────────────────────────────────────────
st.set_page_config(page_title="CityScape Segmentation", layout="wide")
page  = st.sidebar.radio("Navigate", ["Page 1 - Training Metrics", "Page 2 - Inference"])
model, device = load_model()

# ════════════════════════════════════════════════
# PAGE 1
# ════════════════════════════════════════════════
if page == "Page 1 - Training Metrics":
    st.title("CityScape Segmentation - Training Dashboard")
    st.markdown("---")

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        history    = results["history"]
        test_miou  = results["test_miou"]
        test_mdice = results["test_mdice"]

        col1, col2 = st.columns(2)
        col1.metric("Test mIOU",  "%.4f" % test_miou,  delta="Pass (>0.48)" if test_miou  > 0.48 else "Below 0.48")
        col2.metric("Test mDice", "%.4f" % test_mdice, delta="Pass (>0.48)" if test_mdice > 0.48 else "Below 0.48")
        st.markdown("---")

        if os.path.exists(PLOTS_PATH):
            st.subheader("Training Curves")
            st.image(PLOTS_PATH, use_column_width=True)
        else:
            epochs = list(range(1, len(history["loss"]) + 1))
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].plot(epochs, history["loss"],  "b-o", markersize=4); axes[0].set_title("Training Loss")
            axes[1].plot(epochs, history["miou"],  "g-o", markersize=4); axes[1].set_title("mIOU")
            axes[1].axhline(test_miou,  color="r", linestyle="--", label="Test %.4f" % test_miou); axes[1].legend()
            axes[2].plot(epochs, history["mdice"], "m-o", markersize=4); axes[2].set_title("mDice")
            axes[2].axhline(test_mdice, color="r", linestyle="--", label="Test %.4f" % test_mdice); axes[2].legend()
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("results.json not found. Run python train.py first.")

# ════════════════════════════════════════════════
# PAGE 2
# ════════════════════════════════════════════════
else:
    st.title("CityScape Segmentation - Inference")
    st.markdown("Upload **4 test images** to compare ground-truth vs predicted masks.")
    st.markdown("---")

    gt_mask_map = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        for ip, mp in zip(results.get("test_image_paths", []), results.get("test_mask_paths", [])):
            gt_mask_map[os.path.basename(ip)] = mp

    uploaded = st.file_uploader("Upload up to 4 test images", type=["png","jpg","jpeg"],
                                accept_multiple_files=True)
    if uploaded:
        uploaded = uploaded[:4]
        cols = st.columns(len(uploaded))
        for col, uf in zip(cols, uploaded):
            pil_img    = Image.open(uf).convert("RGB")
            pred_mask  = predict(model, device, pil_img)
            pred_color = mask_to_color(pred_mask)
            img_disp   = pil_img.resize((128, 96), Image.NEAREST)

            with col:
                st.markdown("**%s**" % uf.name)
                st.image(img_disp,            caption="Input Image",       use_column_width=True)
                fname = os.path.basename(uf.name)
                if fname in gt_mask_map and os.path.exists(gt_mask_map[fname]):
                    gt = Image.open(gt_mask_map[fname]).convert("RGB").resize((128,96), Image.NEAREST)
                    gt_arr   = np.array(gt)
                    gt_mask  = np.max(gt_arr, axis=-1)
                    gt_color = mask_to_color(gt_mask)
                    st.image(gt_color,         caption="Ground-Truth Mask", use_column_width=True)
                else:
                    st.caption("Ground-truth not found locally")
                st.image(pred_color,           caption="Predicted Mask",    use_column_width=True)
    else:
        st.info("Upload images above to start inference.")