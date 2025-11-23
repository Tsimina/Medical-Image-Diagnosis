import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_configuration.lipschnitz_regularization_config import *
from model_configuration.MobileNetV2_lipschnitz_model import create_lipschitz_mobilenet
from utils.lipschnitz_regularization_metrics import compute_metrics


# Data Loaders
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL, shuffle=False)

    print("Classes:", train_ds.classes)
    return train_loader, val_loader, test_loader, len(train_ds.classes)

# Lipschitz Regularization Term
def lipschitz_penalty(model, x, lambda_lip=0.05):
    """
    Penalizes || grad_x (max softmax prob) ||^2
    """
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    max_prob_sum = probs.max(1)[0].sum()  # sum over batch

    grads = torch.autograd.grad(
        max_prob_sum,
        x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
    return lambda_lip * gp

# Evaluation function
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, prec, rec, f1


# Train one epoch with Lipschitz regularization
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            out = model(x)
            ce_loss = criterion(out, y)
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


# Main training loop
def main():
    train_loader, val_loader, test_loader, n_classes = get_loaders()

    model = create_lipschitz_mobilenet(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, val_loader, criterion
        )

        print(f"\nEpoch {ep:02d}:")
        print(f"  Train: loss={tr_loss:.4f}, acc={tr_acc:.3f}, f1={tr_f1:.3f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.3f}, f1={val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_WEIGHTS)
            print(f"  Saved best model to {BEST_WEIGHTS}")

    # Load best model for testing
    model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)
    test_loss, acc, prec, rec, f1 = evaluate(model, test_loader, criterion)

    print("\n--- TEST RESULTS (Lipschitz MobileNetV2) ---")
    print(f"Loss     : {test_loss:.4f}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")


if __name__ == "__main__":
    main()
