

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import random


MODEL_PATH = r"E:\master an 2\ia3\proiect\Medical-Image-Diagnosis\test_application\saved_models\best_mobilenetv2.pth"    

# ORIGINAL IMAGES ROOT (Normal / Tuberculosis folders)
IMAGE_ROOT = r"E:/master an 2/ia3/proiect/Medical-Image-Diagnosis/test/"  

# NEW: SAVE ATTACKED IMAGES HERE
ATTACKED_ROOT = r"E:\master an 2\ia3\proiect\Medical-Image-Diagnosis\attacked_output"

CSV_PATH = r"E:\master an 2\ia3\proiect\correctly_classified_images_binary.csv"    
ATTACK_PROPORTION = 0.30                
IMG_SIZE = 224

class_names = ["Normal", "Tuberculosis"]
class_to_idx = {c: i for i, c in enumerate(class_names)}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
stds  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)


def load_image(path):
    img = Image.open(path).convert("L")
    x = transform(img).unsqueeze(0)
    return x.to(device)


def apply_one_pixel(x, sol):
    img = x.clone()
    _, _, H, W = img.shape
    rx, ry, rr, rg, rb = sol

    px = int(rx * (W - 1))
    py = int(ry * (H - 1))

    rgb01 = torch.tensor([rr, rg, rb], device=device).view(1,3,1,1)
    rgb_norm = (rgb01 - means) / stds

    img[:, :, py, px] = rgb_norm.view(3)
    return img


def predict(x):
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        return probs.argmax(dim=1).item()


def fitness(sol, x_orig, true_label):
    x_adv = apply_one_pixel(x_orig, sol)
    logits = model(x_adv)
    probs = F.softmax(logits, dim=1)[0]
    return -probs[true_label].item()


def differential_evolution(x_orig, y_true, pop=200, iters=120, F=0.7, CR=1.0):
    dim = 5
    pop_vec = np.random.rand(pop, dim)
    fitness_scores = np.array([fitness(ind, x_orig, y_true) for ind in pop_vec])

    for t in range(iters):
        for i in range(pop):

            idxs = [n for n in range(pop) if n != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)

            mutant = pop_vec[a] + F * (pop_vec[b] - pop_vec[c])
            mutant = np.clip(mutant, 0, 1)

            cross = np.random.rand(dim) < CR
            if not cross.any():
                cross[np.random.randint(0, dim)] = True

            trial = np.where(cross, mutant, pop_vec[i])
            f_trial = fitness(trial, x_orig, y_true)

            if f_trial > fitness_scores[i]:
                pop_vec[i] = trial
                fitness_scores[i] = f_trial

                if predict(apply_one_pixel(x_orig, trial)) != y_true:
                    return trial

    return pop_vec[np.argmax(fitness_scores)]



df = pd.read_csv(CSV_PATH)

num_to_attack = int(len(df) * ATTACK_PROPORTION)
selected_rows = df.sample(n=num_to_attack, random_state=42)

print(f"\n=== Attacking {num_to_attack}/{len(df)} images ===\n")

# Ensure attacked folders exist
for cls in class_names:
    os.makedirs(os.path.join(ATTACKED_ROOT, cls), exist_ok=True)


for idx, row in selected_rows.iterrows():

    filename = row["filename"]
    class_name = row["class"]
    true_label = class_to_idx[class_name]

    source_path = os.path.join(IMAGE_ROOT, class_name, filename)
    save_path = os.path.join(ATTACKED_ROOT, class_name, filename)

    print(f"Attacking: {source_path}")

    x_orig = load_image(source_path)

    solution = differential_evolution(x_orig, true_label)

    x_adv = apply_one_pixel(x_orig, solution)

    # denormalize
    x_adv_denorm = x_adv.clone().squeeze(0)
    for c in range(3):
        x_adv_denorm[c] = x_adv_denorm[c] * stds[0][c] + means[0][c]

    x_adv_denorm = torch.clamp(x_adv_denorm, 0, 1)
    img_np = (x_adv_denorm.permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    adv_img = Image.fromarray(img_np)

    # SAVE INTO ATTACKED FOLDER (NOT overwrite original)
    adv_img.save(save_path)

print("\n✓ DONE! Attacked images saved into:", ATTACKED_ROOT)
