import torch
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

# Compute metrics function
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

# FGSM attack function
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
    
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    # Copy input and set requires_grad
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True
    y = y.to(device)

    # Forward + loss
    out = model(x_adv)
    loss = criterion(out, y)

    # Backward w.r.t. input
    model.zero_grad()
    loss.backward()

    # Sign of the gradient
    grad_sign = x_adv.grad.data.sign()

    # FGSM perturbation
    x_adv = x_adv + epsilon * grad_sign

    # Clamp to normalized limits
    if x_min_tensor is not None and x_max_tensor is not None:
        x_min_tensor = x_min_tensor.to(device)
        x_max_tensor = x_max_tensor.to(device)
        x_adv = torch.max(torch.min(x_adv, x_max_tensor), x_min_tensor)

    return x_adv.detach()

# Evaluate FGSM function
def evaluate_fgsm(
    model: Any,
    loader,
    epsilon: float,
    criterion: Any,
    device: torch.device = None,
    x_min_tensor: torch.Tensor = None,
    x_max_tensor: torch.Tensor = None,
):
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Generate adversarial examples
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

        # Forward on adversarial images
        with torch.no_grad():
            out_adv = model(x_adv)
            loss = criterion(out_adv, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out_adv.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    
    return avg_loss, acc, prec, rec, f1
