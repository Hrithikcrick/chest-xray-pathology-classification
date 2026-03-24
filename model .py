import torch.nn as nn
from torchvision.models import efficientnet_b0, convnext_tiny
import timm


def build_effnet_b0(num_classes: int):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_deit3_small(num_classes: int):
    model = timm.create_model(
        "deit3_small_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )
    return model


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt
