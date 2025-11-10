import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os

# Config
folder_path = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\one_pixel' 
models_dir = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\test_application\saved_models'
arch = "best_efficientnet_b0"   # or "best_efficientnet_b0"
classes = ["NORMAL", "PNEUMONIA"] 
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
    print(f"Loading model from: {model_path}")
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

counts = {c: 0 for c in classes}

for img_name in image_files:
    path = os.path.join(folder_path, img_name)
    img = Image.open(path).convert("L")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
        pred_label = classes[pred]
        counts[pred_label] += 1

print("\n--- Summary ---")
for c in classes:
    print(f"{c}: {counts[c]} images")
