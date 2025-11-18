import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = r'path_to_your_data_directory'  
RESULTS_DIR = r'path_to_your_results_directory'  
BASELINE_WEIGHTS = RESULTS_DIR + r'\best_mobilenetv2.pth'  
DISTILLED_WEIGHTS = RESULTS_DIR + r'\best_mobilenetv2_distilled.pth'

# Imagenet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training hyperparams
EPOCHS_BASELINE = 25
EPOCHS_DISTILL = 25
BATCH_TRAIN = 8
BATCH_EVAL = 16
LR = 1e-3

# Defensive distillation hyperparams
T = 50       
ALPHA = 0.9    
