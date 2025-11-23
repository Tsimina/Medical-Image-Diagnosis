import torch.nn as nn
from torchvision.models import mobilenet_v2

# Model creation function
def create_mobilenet_v2(num_classes: int, weights=None):
    model = mobilenet_v2(weights=weights)  
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
