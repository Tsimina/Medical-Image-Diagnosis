import argparse
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os


# Arguments
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run inference over a folder of images using a saved model")
    p.add_argument("--folder", "-f", dest="folder_path",
                   default=r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\test_set',
                   help="Folder with images to classify")
    p.add_argument("--models-dir", dest="models_dir",
                   default=r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\test_application\saved_models',
                   help="Directory containing saved model .pth files")
    p.add_argument("--model-file", dest="model_file", default=None,
                   help="Explicit model .pth file path (overrides --models-dir and --arch)")
    p.add_argument("--arch", dest="arch", default="best_mobilenetv2",
                   help="Model/filename prefix to load (used when --model-file not provided)")
    p.add_argument("--classes", dest="classes",
                   default="NORMAL,PNEUMONIA",
                   help="Comma-separated class labels in the same order as training (default: NORMAL,PNEUMONIA)")
    p.add_argument("--device", dest="device", default=None,
                   help="Device to use: 'cpu' or 'cuda'. By default auto-detects CUDA if available")
    return p.parse_args(argv)


args = parse_args()
folder_path = args.folder_path
models_dir = args.models_dir
arch = args.arch
classes = [c.strip() for c in args.classes.split(",") if c.strip()]
device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

# Model path
if args.model_file:
    model_path = args.model_file
else:
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

# Load model
try:
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

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
