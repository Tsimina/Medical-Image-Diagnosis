import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model_configuration.deffensive_distilation_config import *
from model_configuration.Mobilenetv2_model import create_mobilenet_v2
from utils.deffensive_distilation_metrics import *


# Data Loaders
def get_loaders():
    train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=eval_tf)


    train_loader = DataLoader(
        train_ds, batch_size=BATCH_TRAIN, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_EVAL, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_EVAL, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    print("Train classes:", train_ds.classes)
    print("Class-to-idx:", train_ds.class_to_idx)

    return train_loader, val_loader, test_loader, len(train_ds.classes)


# Training function for defensive distillation
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(RESULTS_DIR, "metrics_mobilenetv2_distilled.txt")

    # Data loaders
    train_loader, val_loader, test_loader, n_classes = get_loaders()

    # teacher - baseline model pre-trained
    teacher = create_mobilenet_v2(n_classes, weights=None).to(DEVICE)
    teacher.load_state_dict(torch.load(BASELINE_WEIGHTS, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Loaded TEACHER model from:", BASELINE_WEIGHTS)

    # Student - distilled model
    student = create_mobilenet_v2(n_classes, weights=None).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # Training loop
    for ep in range(1, EPOCHS_DISTILL + 1):

        # Training (defensive distillation)
        tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch_distill(
            teacher=teacher,
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            device=DEVICE
        )

        # Validation (clean)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_clean(
            student, val_loader, criterion, device=DEVICE
        )

        # Metrics logging
        print(f"\n[Distill] Epoch {ep:02d}:")
        print(f"  Train loss: {tr_loss:.4f}, acc: {tr_acc:.3f}, "
              f"prec: {tr_prec:.3f}, rec: {tr_rec:.3f}, f1: {tr_f1:.3f}")
        print(f"  Val   loss: {val_loss:.4f}, acc: {val_acc:.3f}, "
              f"prec: {val_prec:.3f}, rec: {val_rec:.3f}, f1: {val_f1:.3f}")

        # -Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), DISTILLED_WEIGHTS)

        if ep % 5 == 0:
            metrics_file = os.path.join(r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\results', "metrics_distilled_mobilenetv2.txt")
            save_metrics_file(metrics_file, ep, (tr_loss, tr_acc, tr_prec, tr_rec, tr_f1), (val_loss, val_acc, val_prec, val_rec, val_f1))
            print(f"  Saved metrics to {metrics_file}")

    print("Metrics log:", metrics_file)

if __name__ == "__main__":
    main()