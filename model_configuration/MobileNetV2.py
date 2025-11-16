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
# Use a non-interactive backend so saving figures works on machines without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    batch_size=16,
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    epochs = 25

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

    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            probs = torch.nn.functional.softmax(out, dim=1)
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_proba.extend(probs.cpu().numpy())  # shape: (batch_size, n_classes)
            y_true.extend(y.cpu().numpy())

    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n--- Test Results ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("Confusion Matrix:\n", cm)

    # Compute ROC curve and AUC (for binary or multiclass)
    y_true_arr = np.array(y_true)
    y_proba_arr = np.array(y_proba)
    print(f"y_proba_arr.shape: {y_proba_arr.shape}")
    n_samples = y_true_arr.shape[0]
    if n_samples == 0:
        print("No test samples found; skipping ROC/AUC computation.")
    else:
        n_classes = y_proba_arr.shape[1] if y_proba_arr.ndim == 2 else 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\results')
        os.makedirs(results_dir, exist_ok=True)
        try:
            if n_classes <= 1:
                print("Insufficient class probability output for ROC (n_classes<=1). Skipping.")
            elif n_classes == 2:
                # Binary classification: use probability for class 1
                y_proba_pos = y_proba_arr[:, 1]
                # Ensure there are both classes present in y_true
                unique_classes = np.unique(y_true_arr)
                if unique_classes.size < 2:
                    print(f"Only one class present in test labels ({unique_classes}); cannot compute ROC.")
                else:
                    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_pos)
                    roc_auc = auc(fpr, tpr)
                    print(f"ROC AUC Score: {roc_auc:.4f}")

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve - MobileNetV2')
                    plt.legend(loc="lower right")
                    plt.grid(alpha=0.3)

                    roc_plot_path = os.path.join(results_dir, f'roc_curve_mobilenetv2_{timestamp}.png')
                    plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
                    print(f"ROC curve saved to {roc_plot_path}")
                    plt.close()
            else:
                # Multiclass: compute one-vs-rest ROC for each class
                classes_idx = list(range(n_classes))
                y_true_bin = label_binarize(y_true_arr, classes=classes_idx)
                if y_true_bin.shape[1] != n_classes:
                    # label_binarize might produce fewer columns if some labels are missing
                    print("Warning: some classes missing in y_true; ROC will be computed for present classes only.")

                # Compute ROC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()

                plt.figure(figsize=(9, 7))
                colors = plt.cm.get_cmap('tab10')
                for i in range(n_classes):
                    try:
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba_arr[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        plt.plot(fpr[i], tpr[i], lw=2, color=colors(i % 10),
                                 label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
                    except ValueError:
                        # Occurs if a class is absent in y_true_bin[:, i]
                        print(f"Skipping ROC for class {i} (no positive samples in y_true)")

                # micro-average ROC
                try:
                    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_proba_arr.ravel())
                    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
                    plt.plot(fpr['micro'], tpr['micro'], label=f"micro-average (AUC = {roc_auc['micro']:.4f})",
                             color='deeppink', linestyle=':', linewidth=3)
                except Exception:
                    pass

                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve - MobileNetV2 (multiclass)')
                plt.legend(loc="lower right", fontsize='small')
                plt.grid(alpha=0.3)

                roc_plot_path = os.path.join(results_dir, f'roc_curve_mobilenetv2_multiclass_{timestamp}.png')
                plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
                print(f"Multiclass ROC curve saved to {roc_plot_path}")
                plt.close()
        except Exception as e:
            print(f"Unexpected error when computing/saving ROC: {e}")
            traceback.print_exc()
        finally:
            # Diagnostic listing
            try:
                print("Results directory listing:")
                for fn in sorted(os.listdir(results_dir)):
                    print("  ", fn)
            except Exception:
                pass


if __name__ == '__main__':
    freeze_support()
    main()
