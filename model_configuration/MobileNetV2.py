import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import os
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
data_dir = r"C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\chest_xray"   #train/ val/ test 

# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True

# Transforms (X-rays -> 3 channels)
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

# Datasets and loaders
train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_tf)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,  
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Classes:", train_ds.classes)
print("Class-to-idx:", train_ds.class_to_idx)

# Model, loss, optimizer, scaler
model = mobilenet_v2(weights=None)
n_classes = len(train_ds.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, n_classes)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())

# Metrics
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


def save_metrics_file(path, epoch, train_metrics, val_metrics):
    exists = os.path.exists(path)
    
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("=" * 80 + "\n\n")
        
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_metrics
        val_loss, val_acc, val_prec, val_rec, val_f1 = val_metrics
        
        # Write epoch header and metrics in a readable format
        f.write(f"Epoch {epoch:03d}\n")
        f.write("-" * 40 + "\n")
        f.write("Training Metrics:\n")
        f.write(f"  Loss:      {tr_loss:.6f}\n")
        f.write(f"  Accuracy:  {tr_acc:.6f}\n")
        f.write(f"  Precision: {tr_prec:.6f}\n")
        f.write(f"  Recall:    {tr_rec:.6f}\n")
        f.write(f"  F1 Score:  {tr_f1:.6f}\n\n")
        
        f.write("Validation Metrics:\n")
        f.write(f"  Loss:      {val_loss:.6f}\n")
        f.write(f"  Accuracy:  {val_acc:.6f}\n")
        f.write(f"  Precision: {val_prec:.6f}\n")
        f.write(f"  Recall:    {val_rec:.6f}\n")
        f.write(f"  F1 Score:  {val_f1:.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).detach().cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, prec, rec, f1

# Training 
def main():
    best_val_acc, best_path = 0.0, "best_mobilenetv2.pth"
    epochs = 30

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(val_loader)

        print(f"\nEpoch {ep:02d}:")
        print(f"  Train  loss: {tr_loss:.4f}, acc: {tr_acc:.3f}, prec: {tr_prec:.3f}, rec: {tr_rec:.3f}, f1: {tr_f1:.3f}")
        print(f"  Val  loss: {val_loss:.4f}, acc: {val_acc:.3f}, prec: {val_prec:.3f}, rec: {val_rec:.3f}, f1: {val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        # Save metrics 
        if ep % 10 == 0:
            metrics_file = os.path.join(r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\results', "metrics_mobilenetv2.txt")
            save_metrics_file(metrics_file, ep, (tr_loss, tr_acc, tr_prec, tr_rec, tr_f1), (val_loss, val_acc, val_prec, val_rec, val_f1))
            print(f"  Saved metrics to {metrics_file}")

    # Eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Test Results ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("Confusion Matrix:\n", cm)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
