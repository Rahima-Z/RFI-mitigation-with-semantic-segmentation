
import torch
import torch.nn as nn
import torch.nn.functional as F


# Losses individuelles

def ce_loss(logits, target, class_weights=None, **kwargs):
    return F.cross_entropy(logits, target, weight=class_weights)


def dice_loss(logits, target, num_classes=2, eps=1e-6, **kwargs):
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_onehot, dims)
    cardinality = torch.sum(probs + target_onehot, dims)

    dice_per_class = (2. * intersection + eps) / (cardinality + eps)
    return 1. - dice_per_class.mean()


def focal_loss(logits, target, alpha=0.25, gamma_focal=2.0, **kwargs):
    ce = F.cross_entropy(logits, target, reduction="none")
    p_t = torch.exp(-ce)
    focal = alpha * (1 - p_t) ** gamma_focal * ce
    return focal.mean()


def tversky_loss(logits, target, num_classes=2, alpha=0.5, beta=0.5, eps=1e-6, **kwargs):

    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    tp = torch.sum(probs * target_onehot, dims)
    fp = torch.sum(probs * (1 - target_onehot), dims)
    fn = torch.sum((1 - probs) * target_onehot, dims)

    tversky_per_class = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1. - tversky_per_class.mean()


def focal_tversky_loss(logits, target, num_classes=2, alpha=0.5, beta=0.5, gamma_ft=1.33, eps=1e-6, **kwargs):

    tv = tversky_loss(logits, target, num_classes, alpha, beta, eps)
    return tv ** (1.0 / gamma_ft)


# Liste des losses disponibles

LOSS_REGISTRY = {
    "ce": ce_loss,
    "dice": dice_loss,
    "focal": focal_loss,
    "tversky": tversky_loss,
    "focal_tversky": focal_tversky_loss,
}



# Combinaisons pondérées de losses

def build_combined_loss(components, **loss_kwargs):

    for name in components:
        if name not in LOSS_REGISTRY:
            raise ValueError(f"Loss inconnue: {name}. Options: {list(LOSS_REGISTRY.keys())}")

    def combined(logits, target):
        total = 0.0
        for name, weight in components.items():
            fn = LOSS_REGISTRY[name]
            total = total + weight * fn(logits, target, **loss_kwargs)
        return total

    return combined
