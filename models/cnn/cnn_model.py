"""
models/cnn/cnn_model.py
------------------------
CNN baseline model for deepfake detection.

Uses transfer learning from EfficientNet-B4 (default),
EfficientNet-B0, MobileNetV2, or Xception.

The CNN learns LOCAL facial artifacts:
  - Blending artifacts at face boundaries
  - Texture inconsistencies in skin
  - Compression artifacts
  - Unnatural high-frequency patterns

Architecture:
  Pretrained CNN Backbone
       ↓
  Global Average Pooling
       ↓
  Dropout + FC (512)
       ↓
  BatchNorm + ReLU
       ↓
  Dropout + FC (2)
       ↓
  Real / Fake prediction

Usage:
    from models.cnn.cnn_model import CNNModel
    model = CNNModel(backbone='efficientnet_b4', pretrained=True, num_classes=2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision import models as tv_models
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False


# ── Supported backbone configurations ─────────────────────────────────────────
BACKBONE_CONFIG = {
    # timm names → feature dimensions
    "efficientnet_b0":  {"timm": "efficientnet_b0",        "out_dim": 1280},
    "efficientnet_b4":  {"timm": "efficientnet_b4",        "out_dim": 1792},
    "efficientnet_b7":  {"timm": "efficientnet_b7",        "out_dim": 2560},
    "mobilenet_v2":     {"timm": "mobilenetv2_100",        "out_dim": 1280},
    "mobilenet_v3":     {"timm": "mobilenetv3_large_100",  "out_dim": 960},
    "xception":         {"timm": "xception",               "out_dim": 2048},
    "resnet50":         {"timm": "resnet50",               "out_dim": 2048},
    "convnext_tiny":    {"timm": "convnext_tiny",          "out_dim": 768},
}


class CNNModel(nn.Module):
    """
    CNN-based deepfake detector with a pretrained backbone.

    Args:
        backbone:     Name of backbone (see BACKBONE_CONFIG).
        pretrained:   Load ImageNet pretrained weights.
        num_classes:  2 for binary (real/fake).
        dropout_rate: Dropout applied before classification head.
        hidden_dim:   Hidden layer dimension in the classifier head.
        freeze_backbone: Freeze backbone weights (feature extractor only mode).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # ── Load backbone ──────────────────────────────────────────────────────
        self.backbone, feat_dim = self._build_backbone(backbone, pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Classification head ────────────────────────────────────────────────
        self.classifier = self._build_head(feat_dim, hidden_dim, num_classes, dropout_rate)

        # ── Weight initialization ──────────────────────────────────────────────
        self._init_head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W).
        Returns:
            Logits tensor (B, num_classes).
        """
        features = self.backbone(x)     # (B, feat_dim)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features (for Hybrid model integration)."""
        return self.backbone(x)

    # ── Builder helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Load a timm backbone with pooling, no classifier head."""
        if backbone_name not in BACKBONE_CONFIG:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(BACKBONE_CONFIG.keys())}"
            )

        cfg = BACKBONE_CONFIG[backbone_name]

        if TIMM_AVAILABLE:
            model = timm.create_model(
                cfg["timm"],
                pretrained=pretrained,
                num_classes=0,       # Remove classifier
                global_pool="avg",   # Global average pooling
            )
            # Get actual output dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = model(dummy)
                feat_dim = out.shape[1]
        elif TV_AVAILABLE:
            # torchvision fallback (only supports some models)
            model, feat_dim = CNNModel._torchvision_backbone(backbone_name, pretrained)
        else:
            raise ImportError("Install either 'timm' or 'torchvision': pip install timm")

        return model, feat_dim

    @staticmethod
    def _torchvision_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Torchvision fallback for when timm is unavailable."""
        weights = "IMAGENET1K_V1" if pretrained else None
        if "efficientnet_b0" in name:
            m = tv_models.efficientnet_b0(weights=weights)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
        elif "mobilenet_v2" in name:
            m = tv_models.mobilenet_v2(weights=weights)
            feat_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
        elif "resnet" in name:
            m = tv_models.resnet50(weights=weights)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
        else:
            # Default to ResNet50
            m = tv_models.resnet50(weights=weights)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
        return m, feat_dim

    @staticmethod
    def _build_head(
        in_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Inference helpers ──────────────────────────────────────────────────────
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return (predicted_class, confidence).

        Args:
            x: Input tensor (B, C, H, W).
        Returns:
            (predicted_labels [B], probabilities [B, 2])
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


# ── Factory function ───────────────────────────────────────────────────────────
def build_cnn_model(config) -> CNNModel:
    """
    Build CNNModel from a config object (DotDict or similar).

    Args:
        config: Config object with .model.* attributes.
    Returns:
        CNNModel instance.
    """
    return CNNModel(
        backbone=config.model.cnn_backbone,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
    )


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CNNModel...")
    model = CNNModel(backbone="efficientnet_b4", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    preds, probs = model.predict(x)
    params = model.count_parameters()
    print(f"Input:    {x.shape}")
    print(f"Logits:   {logits.shape}")
    print(f"Probs:    {probs.shape}")
    print(f"Params:   {params['total']:,} total / {params['trainable']:,} trainable")
    print("CNNModel OK!")
