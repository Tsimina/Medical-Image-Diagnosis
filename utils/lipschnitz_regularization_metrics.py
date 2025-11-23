import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix
)

# Metrics computation
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return acc, prec, rec, f1

# Evaluation 
@torch.no_grad()
def evaluate_clean(model, loader, criterion, device):

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
