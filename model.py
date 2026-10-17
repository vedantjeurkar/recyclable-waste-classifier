import torch.nn as nn
from torchvision import models


def build_model(num_classes=6, pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze the pretrained part so its learned "vision" knowledge is kept
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final layer: 1000 ImageNet classes -> our 6 waste classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model