import torch

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
DATA_DIR = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\chest_xray'
RESULTS_DIR = r'C:\Users\simin\OneDrive\Desktop\Master_an_2\IA3\lab\Medical-Image-Diagnosis\test_application\saved_models'
BASELINE_WEIGHTS = RESULTS_DIR + r'\best_mobilenetv2.pth'  # use existing trained model
DISTILLED_WEIGHTS = RESULTS_DIR + r'\best_mobilenetv2_distilled.pth'

# imagenet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# training hyperparams
EPOCHS_BASELINE = 25
EPOCHS_DISTILL = 20
BATCH_TRAIN = 16
BATCH_EVAL = 32
LR = 1e-3

# defensive distillation params
T = 10.0         # temperatură
ALPHA = 0.7      # ponderare soft vs hard loss
