import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os

# Metrics computation
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec, f1


# Save metrics to file
def save_metrics_file(path, epoch, train_metrics, val_metrics):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("=" * 80 + "\n\n")

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_metrics
        val_loss, val_acc, val_prec, val_rec, val_f1 = val_metrics

        f.write(f"Epoch {epoch:03d}\n")
        f.write("-" * 40 + "\n")

        f.write("Training Metrics:\n")
        f.write(f"  Loss: {tr_loss:.6f}\n")
        f.write(f"  Acc : {tr_acc:.6f}\n")
        f.write(f"  Prec: {tr_prec:.6f}\n")
        f.write(f"  Rec : {tr_rec:.6f}\n")
        f.write(f"  F1  : {tr_f1:.6f}\n\n")

        f.write("Validation Metrics:\n")
        f.write(f"  Loss: {val_loss:.6f}\n")
        f.write(f"  Acc : {val_acc:.6f}\n")
        f.write(f"  Prec: {val_prec:.6f}\n")
        f.write(f"  Rec : {val_rec:.6f}\n")
        f.write(f"  F1  : {val_f1:.6f}\n\n")

        f.write("=" * 80 + "\n\n")


# Confusion Matrix Plotting
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# Plot ROC and compute AUC
def plot_roc_auc(y_true, y_proba, results_dir, prefix="baseline"):
    os.makedirs(results_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    n_classes = y_proba.shape[1]

    # Plot ROC for binary classification
    y_proba_pos = y_proba[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    save_path = os.path.join(results_dir, f"roc_{prefix}_{timestamp}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    return roc_auc, save_path