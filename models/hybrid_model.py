"""Hybrid ResNet50 + DenseNet121 model used for crop disease/damage classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class HybridResNetDenseNet(nn.Module):
    """ResNet50 + DenseNet121 feature fusion with SE attention and MLP head."""

    def __init__(self, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.4):
        super().__init__()

        res_model = resnet50(weights=None)
        self.resnet_backbone = nn.Sequential(*list(res_model.children())[:-1])

        den_model = densenet121(weights=None)
        self.densenet_backbone = den_model.features
        self.densenet_pool = nn.AdaptiveAvgPool2d((1, 1))

        fused_dim = 2048 + 1024
        self.fusion_norm = nn.BatchNorm1d(fused_dim)
        self.se_block = SEBlock(fused_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        res_feat = self.resnet_backbone(x).flatten(1)
        den_feat = F.relu(self.densenet_backbone(x), inplace=True)
        den_feat = self.densenet_pool(den_feat).flatten(1)
        fused = torch.cat([res_feat, den_feat], dim=1)
        fused = self.se_block(self.fusion_norm(fused))
        return self.classifier(fused)


def load_checkpoint(path: str, device: torch.device):
    """Load the trained checkpoint saved by the notebook."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names")
        hidden_dim = int(checkpoint.get("hidden_dim", 1024))
        dropout = float(checkpoint.get("dropout", 0.4))
    else:
        state_dict = checkpoint
        class_names = None
        hidden_dim = 1024
        dropout = 0.4

    if class_names is None:
        raise ValueError("Checkpoint does not contain class_names metadata.")

    model = HybridResNetDenseNet(
        num_classes=len(class_names),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, class_names
