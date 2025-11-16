import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from utils.FGSM import fgsm_attack, evaluate_fgsm, compute_metrics 
from multiprocessing import freeze_support

# ======================= CONFIG ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = r"C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\chest_xray"
weights_path = r"C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\test_application\saved_models\best_mobilenetv2.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# limitele echivalente [0,1] în spațiul normalizat
IMAGENET_MIN = [(0.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
IMAGENET_MAX = [(1.0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]


# ======================= DATA ============================
def get_test_loader(batch_size=32):
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("Classes:", test_ds.classes)
    print("Class-to-idx:", test_ds.class_to_idx)
    return test_loader, len(test_ds.classes)


# ======================= MODEL ===========================
def load_mobilenetv2(weights_path, num_classes):
    # 1. Creezi modelul cu arhitectura corectă
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,  
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ======================= CLEAN EVAL ======================
@torch.no_grad()
def evaluate_clean(model, loader, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
    return avg_loss, acc, prec, rec, f1


# ======================= MAIN ============================
def main():
    # 1) data
    test_loader, n_classes = get_test_loader(batch_size=32)

    # 2) model
    model = load_mobilenetv2(weights_path, num_classes=n_classes)
    criterion = nn.CrossEntropyLoss()

    # 3) tensori pentru clamp în spațiul normalizat
    x_min_tensor = torch.tensor(IMAGENET_MIN, device=device).view(1, 3, 1, 1)
    x_max_tensor = torch.tensor(IMAGENET_MAX, device=device).view(1, 3, 1, 1)

    # 4) evaluare pe imagini curate
    clean_loss, clean_acc, clean_prec, clean_rec, clean_f1 = evaluate_clean(
        model=model,
        loader=test_loader,
        criterion=criterion
    )
    print("\n=== Clean Test Evaluation ===")
    print(f"Loss: {clean_loss:.4f}, "
          f"Acc: {clean_acc:.3f}, "
          f"Prec: {clean_prec:.3f}, "
          f"Rec: {clean_rec:.3f}, "
          f"F1: {clean_f1:.3f}")

    # 5) evaluare FGSM pentru mai multe epsilons
    epsilons = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]

    print("\n=== FGSM Attack Evaluation (test set) ===")
    for eps in epsilons:
        adv_loss, adv_acc, adv_prec, adv_rec, adv_f1 = evaluate_fgsm(
            model=model,
            loader=test_loader,
            epsilon=eps,
            criterion=criterion,
            device=device,
            x_min_tensor=x_min_tensor,
            x_max_tensor=x_max_tensor
        )

        print(f"Epsilon = {eps:.3f} -> "
              f"Loss: {adv_loss:.4f}, "
              f"Acc: {adv_acc:.3f}, "
              f"Prec: {adv_prec:.3f}, "
              f"Rec: {adv_rec:.3f}, "
              f"F1: {adv_f1:.3f}")


if __name__ == "__main__":
    freeze_support()
    main()
