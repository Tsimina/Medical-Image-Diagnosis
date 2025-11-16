# utils.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from model_configuration.deffensive_distilation_config import (
    DEVICE, T, ALPHA
)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    Calculează Accuracy, Precision, Recall, F1 (macro).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, prec, rec, f1


def save_metrics_file(path, epoch, train_metrics, val_metrics):
    """
    Salvează metricile pentru un epoch (Train + Val) într-un fișier .txt.
    Format similar cu scriptul tău original.
    """
    exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("=" * 80 + "\n\n")

        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_metrics
        val_loss, val_acc, val_prec, val_rec, val_f1 = val_metrics

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


# ---------------------------------------------------------
# CLEAN EVAL
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_clean(model, loader, criterion, device=DEVICE):
    """
    Evaluează un model pe loader (fără atac), returnează:
    loss, acc, prec, rec, f1
    """
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

    return avg_loss, acc, prec, rec, f1


# ---------------------------------------------------------
# DEFENSIVE DISTILLATION – UN SINGUR EPOCH
# ---------------------------------------------------------
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available())
_ce_loss = nn.CrossEntropyLoss()
_kl_div  = nn.KLDivLoss(reduction="batchmean")


def train_one_epoch_distill(teacher, student, loader, optimizer, device=DEVICE):
    """
    Un epoch de defensive distillation.
    teacher: model înghețat (eval, fără grad)
    student: model antrenat
    """
    teacher.eval()
    student.train()

    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # Teacher logits (fără gradient)
        with torch.no_grad():
            teacher_logits = teacher(x)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            enabled=torch.cuda.is_available()
        ):
            student_logits = student(x)

            # soft labels de la teacher + student
            teacher_soft    = torch.softmax(teacher_logits / T, dim=1)
            student_log_soft = torch.log_softmax(student_logits / T, dim=1)

            # pierdere de distillation (KL)
            loss_soft = _kl_div(student_log_soft, teacher_soft) * (T * T)

            # pierdere clasică față de label-uri
            loss_hard = _ce_loss(student_logits, y)

            # combinația finală
            loss = ALPHA * loss_soft + (1.0 - ALPHA) * loss_hard

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(student_logits.argmax(1).detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

    return avg_loss, acc, prec, rec, f1
