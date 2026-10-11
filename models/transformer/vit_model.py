"""
models/transformer/vit_model.py
---------------------------------
Vision Transformer (ViT) baseline for deepfake detection.

Models global spatial relationships between facial patch regions,
capturing long-range dependencies that CNNs miss (e.g., symmetry
violations, global lighting inconsistencies across face regions).

Two options:
  1. Full ViT from scratch (smaller, for limited compute).
  2. Pretrained ViT via timm (recommended).

Architecture:
  Input Image (224x224x3)
       ↓
  Patch Embedding (16x16 patches → 196 tokens)
       ↓
  [CLS] Token Prepended → 197 tokens
       ↓
  Position Embedding Added
       ↓
  Transformer Encoder × L layers
    (Multi-head Self-Attention + MLP + LayerNorm)
       ↓
  CLS Token Output
       ↓
  Classification Head
       ↓
  Real / Fake

Usage:
    from models.transformer.vit_model import ViTModel
    model = ViTModel(pretrained=True, model_name='vit_base_patch16_224')
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Option 1 — Custom ViT (from scratch, no timm dependency)
# ══════════════════════════════════════════════════════════════════════════════
class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and embed each patch.

    Args:
        image_size:  Input image size (assumes square).
        patch_size:  Size of each patch (square).
        in_channels: Input channels (3 for RGB).
        embed_dim:   Embedding dimension per patch.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        )
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, n_patches, embed_dim)
        """
        x = self.proj(x)                # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)               # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)          # (B, n_patches, embed_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block:
      LayerNorm → Multi-Head Attention → Residual
      LayerNorm → MLP → Residual

    Args:
        embed_dim:  Model dimension.
        num_heads:  Number of attention heads.
        mlp_ratio:  MLP hidden dim = embed_dim * mlp_ratio.
        dropout:    Dropout rate.
        attn_drop:  Attention dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)

        # MLP with residual
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

    def get_attention_weights(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output and attention weights (for visualization)."""
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, attn_weights


class CustomViT(nn.Module):
    """
    Full Vision Transformer built from scratch.
    Suitable for smaller datasets or when timm is unavailable.

    Args:
        image_size:    Input image size.
        patch_size:    Patch size (16 is standard).
        in_channels:   Input channels.
        num_classes:   Output classes.
        embed_dim:     Token embedding dimension.
        num_heads:     Attention heads.
        num_layers:    Number of transformer blocks.
        mlp_ratio:     MLP expansion ratio.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        x = self.patch_embed(x)   # (B, n_patches, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, n_patches+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]          # CLS token output
        return self.head(cls_out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features (for Hybrid model)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def get_attention_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return logits and all-layer attention weights for visualization."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        all_attns = []
        for block in self.blocks:
            x, attn = block.get_attention_weights(x)
            all_attns.append(attn)

        x = self.norm(x)
        return self.head(x[:, 0]), all_attns


# ══════════════════════════════════════════════════════════════════════════════
# Option 2 — Pretrained ViT via timm (recommended)
# ══════════════════════════════════════════════════════════════════════════════
class PretrainedViTModel(nn.Module):
    """
    ViT model using pretrained timm backbone.

    Recommended models:
      - vit_base_patch16_224     (86M params, best quality)
      - vit_small_patch16_224    (22M params, balanced)
      - vit_tiny_patch16_224     (5.7M params, fast)
      - deit_small_patch16_224   (DeiT, data-efficient)
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # Remove classifier
        )

        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs


# ══════════════════════════════════════════════════════════════════════════════
# Unified ViTModel (picks best available option)
# ══════════════════════════════════════════════════════════════════════════════
class ViTModel(nn.Module):
    """
    Unified ViT model that selects pretrained (timm) or custom implementation.

    Args:
        pretrained:      Use pretrained timm backbone (True) or custom (False).
        model_name:      timm model name (used only if pretrained=True).
        image_size:      Input image size.
        num_classes:     Output classes.
        embed_dim:       Embedding dim (used only for custom ViT).
        num_heads:       Attention heads (custom ViT only).
        num_layers:      Encoder blocks (custom ViT only).
        dropout_rate:    Dropout probability.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "vit_small_patch16_224",
        image_size: int = 224,
        num_classes: int = 2,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if pretrained and TIMM_AVAILABLE:
            self.model = PretrainedViTModel(
                model_name=model_name,
                pretrained=True,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
            )
            self._mode = "pretrained"
        else:
            self.model = CustomViT(
                image_size=image_size,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
            )
            self._mode = "custom"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(x)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "mode": self._mode}


# ── Factory ────────────────────────────────────────────────────────────────────
def build_vit_model(config) -> ViTModel:
    """Build ViTModel from config."""
    return ViTModel(
        pretrained=config.model.pretrained,
        image_size=config.data.image_size,
        num_classes=config.model.num_classes,
        embed_dim=config.model.transformer.embed_dim,
        num_heads=config.model.transformer.num_heads,
        num_layers=config.model.transformer.num_layers,
        mlp_ratio=config.model.transformer.mlp_ratio,
        dropout_rate=config.model.transformer.dropout,
    )


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing ViTModel (custom)...")
    model = ViTModel(pretrained=False, embed_dim=256, num_heads=4, num_layers=4)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    preds, probs = model.predict(x)
    params = model.count_parameters()
    print(f"Input:    {x.shape}")
    print(f"Logits:   {logits.shape}")
    print(f"Params:   {params['total']:,} total")
    print(f"Mode:     {params['mode']}")
    print("ViTModel OK!")
