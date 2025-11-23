import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r'path_to_your_data_directory'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BATCH_TRAIN = 16
BATCH_EVAL = 32
EPOCHS = 25
LR = 1e-3

BEST_WEIGHTS = "best_mobilenetv2_lipschitz.pth"
