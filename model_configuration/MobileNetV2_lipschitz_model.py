import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch

def apply_spectral_norm(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    return m


def create_lipschitz_mobilenet(num_classes):
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.apply(apply_spectral_norm)
    return model
