import torch
import torch.nn as nn
import csv
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os

# Config
folder_path = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\test_set' 
models_dir = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\test_application\saved_models'
arch = "best_efficientnet_b0"   
classes = ["NORMAL", "PNEUMONIA"] 
csv_path = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\results\true_labels.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
model_path = os.path.join(models_dir, f"{arch}.pth")

# Preprocessing 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Model 
if arch == "best_mobilenetv2":
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
else:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

# load model
try:
    state = torch.load(model_path, map_location=device, weights_only=True)  
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state)
    model = model.to(device).eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Inference
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Load true labels CSV if available (expects filename,label)
true_map = {}
if os.path.exists(csv_path):
    try:
        with open(csv_path, newline='') as tf:
            reader = csv.reader(tf)
            # skip header if present
            first = next(reader, None)
            if first and (first[0].lower() == 'filename' or first[1].lower() == 'label'):
                pass
            else:
                # first row was data
                if first and len(first) >= 2:
                    true_map[first[0]] = first[1]
            for row in reader:
                if not row:
                    continue
                if len(row) >= 2:
                    true_map[row[0]] = row[1]
    except Exception as e:
        print(f"Warning: failed reading true labels CSV {csv_path}: {e}")

results = []
counts = {c: 0 for c in classes}

for img_name in image_files:
    path = os.path.join(folder_path, img_name)
    img = Image.open(path).convert("L")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = probs.max(dim=1)
        pred_idx = int(pred.item())
        pred_label = classes[pred_idx]
        confidence = float(conf.item())
        counts[pred_label] += 1

        true_raw = true_map.get(img_name, '')
        true_label_name = ''
        if true_raw != '':
            try:
                ti = int(true_raw)
                if 0 <= ti < len(classes):
                    true_label_name = classes[ti]
            except Exception:
                true_label_name = str(true_raw)

        correct = ''
        if true_label_name:
            correct = (true_label_name == pred_label)

        results.append((img_name, true_label_name, pred_label, f"{confidence:.4f}", correct))

# Write predictions vs true labels to CSV
out_csv = os.path.join(os.getcwd(), "predictions_vs_true_efficientnet_b0.csv")
with open(out_csv, 'w', newline='', encoding='utf-8') as of:
    w = csv.writer(of)
    w.writerow(["filename", "true_label", "pred_label", "confidence", "result"])
    for r in results:
        w.writerow(r)

print("\n--- Summary ---")
for c in classes:
    print(f"{c}: {counts[c]} images")
