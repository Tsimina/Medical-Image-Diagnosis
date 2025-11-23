import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import os
import random

from attacks.One_Pixel_attack import (
    differential_evolution,
    apply_one_pixel
)

MODEL_PATH = r'path_to_your_model_weights.pth'    

IMAGE_ROOT = r'path_to_your_data_directory/test/' 

ATTACKED_ROOT = r'path_to_your_attacked_output_directory'

CSV_PATH = r'path_to_your_csv_file.csv'  
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


df = pd.read_csv(CSV_PATH)

num_to_attack = int(len(df) * ATTACK_PROPORTION)
selected_rows = df.sample(n=num_to_attack, random_state=42)

print(f"\n=== Attacking {num_to_attack}/{len(df)} images ===\n")

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

    solution = differential_evolution(
        x_orig=x_orig,
        y_true=true_label,
        model=model,
        means=means,
        stds=stds,
        device=device
    )

    x_adv = apply_one_pixel(x_orig, solution, means, stds, device)

    x_adv_denorm = x_adv.clone().squeeze(0)
    for c in range(3):
        x_adv_denorm[c] = x_adv_denorm[c] * stds[0][c] + means[0][c]

    x_adv_denorm = torch.clamp(x_adv_denorm, 0, 1)
    img_np = (x_adv_denorm.permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    adv_img = Image.fromarray(img_np)

    adv_img.save(save_path)

print("\n Saved into:", ATTACKED_ROOT)
