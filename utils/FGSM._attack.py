import torch
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


def fgsm_attack(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: Any,
    device: torch.device = None,
    x_min_tensor: torch.Tensor = None,
    x_max_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """
    x: batch de imagini DEJA normalizate (cum ies din DataLoader)
    y: etichetele corecte
    epsilon: mărimea perturbării în spațiul normalizat
    """
    model.eval()
    # determine device if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    # facem o copie a inputului și o marcăm cu requires_grad
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    y = y.to(device)

    # forward + loss
    out = model(x_adv)
    loss = criterion(out, y)

    # backward față de intrare
    model.zero_grad()
    loss.backward()

    # semnul gradientului
    grad_sign = x_adv.grad.data.sign()

    # aplicăm perturbarea FGSM
    x_adv = x_adv + epsilon * grad_sign

    # clamp în limitele normalizate (echivalent cu [0,1] înainte de Normalize)
    if x_min_tensor is not None and x_max_tensor is not None:
        x_min_tensor = x_min_tensor.to(device)
        x_max_tensor = x_max_tensor.to(device)
        x_adv = torch.max(torch.min(x_adv, x_max_tensor), x_min_tensor)

    return x_adv.detach()


def evaluate_fgsm(
    model: Any,
    loader,
    epsilon: float,
    criterion: Any,
    device: torch.device = None,
    x_min_tensor: torch.Tensor = None,
    x_max_tensor: torch.Tensor = None,
):
    """
    Evaluează modelul pe exemple FGSM generate din setul dat de loader.
    Returnează: loss, acc, prec, rec, f1
    """
    model.eval()

    # determine device if not provided
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # generăm exemple adversariale pentru acest batch
        x_adv = fgsm_attack(
            model=model,
            x=x,
            y=y,
            epsilon=epsilon,
            criterion=criterion,
            device=device,
            x_min_tensor=x_min_tensor,
            x_max_tensor=x_max_tensor,
        )

        # forward pe imaginile adversariale (fără gradient, doar pentru evaluare)
        with torch.no_grad():
            out_adv = model(x_adv)
            loss = criterion(out_adv, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out_adv.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    
    return avg_loss, acc, prec, rec, f1
