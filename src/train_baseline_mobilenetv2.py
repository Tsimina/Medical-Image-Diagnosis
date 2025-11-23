import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_configuration.baseline_config import *
from model_configuration.Mobilenetv2_model import create_mobilenet_v2
from utils.baseline_metrics import compute_metrics, save_metrics_file

def get_loaders():
    train_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", eval_tf)
    test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_EVAL, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_ds.classes)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return (avg_loss, *compute_metrics(y_true, y_pred))

def main():
    train_loader, val_loader, test_loader, n_classes = get_loaders()

    model = create_mobilenet_v2(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(1).detach().cpu().numpy())

        tr_loss = total_loss / len(train_loader.dataset)
        tr_acc, tr_prec, tr_rec, tr_f1 = compute_metrics(y_true, y_pred)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion)

        print(f"\nEpoch {ep:02d}:")
        print(f"  Train  loss: {tr_loss:.4f}, acc: {tr_acc:.3f}, prec: {tr_prec:.3f}, rec: {tr_rec:.3f}, f1: {tr_f1:.3f}")
        print(f"  Val  loss: {val_loss:.4f}, acc: {val_acc:.3f}, prec: {val_prec:.3f}, rec: {val_rec:.3f}, f1: {val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BASELINE_WEIGHTS)

        metrics_file = f"{RESULTS_DIR}/metrics_baseline.txt"
        save_metrics_file(metrics_file, ep,
                          (tr_loss, tr_acc, tr_prec, tr_rec, tr_f1),
                          (val_loss, val_acc, val_prec, val_rec, val_f1))

    print(f"Best model saved to: {BASELINE_WEIGHTS}")

if __name__ == "__main__":
    main()
