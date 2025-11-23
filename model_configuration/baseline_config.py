import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
DATA_DIR = r'path_to_your_train_data_directory'

# Results paths
RESULTS_DIR = r'Path_to_your_results_directory'
BASELINE_WEIGHTS = RESULTS_DIR + r'path_to_baseline_model_weights.pth'

# Hyperparameters
BATCH_TRAIN = 16
BATCH_EVAL = 32
LR = 1e-3
EPOCHS = 25

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
