import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image

# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & Model Loading
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = dataset.classes

model = models.resnet18(weights=None) 
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==========================
# Evaluation Loop
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# Confusion Matrix & Plotting
# ==========================
cm = confusion_matrix(all_labels, all_preds)

def save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Class-wise Confusion Matrix')
    
    # Save as PDF
    plt.savefig("./confusion_matrix.png", bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.pdf'")
    plt.close()

save_confusion_matrix(cm, class_names)

# ==========================
# Class-wise Accuracy Report
# ==========================
with np.errstate(divide='ignore', invalid='ignore'):
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\n" + "="*40)
print(f"{'CLASS':<10} | {'ACCURACY':<10}")
print("-" * 40)
for i, name in enumerate(class_names):
    acc = class_accuracy[i] if not np.isnan(class_accuracy[i]) else 0.0
    print(f"{name:<10} | {acc*100:>8.2f}%")
print("="*40)

# ==========================
# Specific Image Prediction
# ==========================
def predict_specific(image_path):
    if os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img_t)
            conf, pred = torch.max(torch.softmax(out, dim=1), 1)
        print(f"\nTarget Image: {image_path}")
        print(f"Prediction: {class_names[pred.item()]} ({conf.item()*100:.2f}%)")

predict_specific("data/test/5/340.png")