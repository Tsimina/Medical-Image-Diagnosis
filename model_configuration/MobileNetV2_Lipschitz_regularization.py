import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import os
from pathlib import Path
import traceback
from datetime import datetime
from multiprocessing import freeze_support
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
data_dir = r"E:\master an 2\ia3\proiect\Lung Disease Dataset"



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

print("Classes:", train_ds.classes)



### Load model
model = mobilenet_v2(weights=None)
n_classes = len(train_ds.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, n_classes)

# Apply Spectral Normalization
def apply_spectral_norm(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    return m

model = model.apply(apply_spectral_norm)   

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler(enabled=torch.cuda.is_available())



# Lipschitz Regularization Term
def lipschitz_penalty(model, x, lambda_lip=0.05):
    # Penalizes || gradient_x (model output) ||^2

    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    max_logit = probs.max(1)[0].sum()

    grads = torch.autograd.grad(
        max_logit,
        x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()

    return lambda_lip * gp



def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, prec, rec, f1



def train_one_epoch(loader):
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            out = model(x)
            ce_loss = criterion(out, y)

            #
            lip_loss = lipschitz_penalty(model, x, lambda_lip=0.05)

            loss = ce_loss + lip_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, prec, rec, f1



def main():
    best_val_acc = 0.0
    best_path = "best_mobilenetv2_lipschitz.pth"
    epochs = 25

    for ep in range(1, epochs + 1):

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(val_loader)

        print(f"\nEpoch {ep:02d}:")
        print(f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.3f}, f1={tr_f1:.3f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.3f}, f1={val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # Evaluate on test set
    model.load_state_dict(torch.load(best_path))
    model.eval()

    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_proba.extend(probs.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    print("\n--- TEST RESULTS (Lipschitz MobileNetV2) ---")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1-score :", f1)

if __name__ == '__main__':
    freeze_support()
    main()
