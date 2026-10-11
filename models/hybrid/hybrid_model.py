"""
models/hybrid/hybrid_model.py
-------------------------------
Hybrid CNN–Transformer model for deepfake detection.

This is the MAIN proposed architecture.

Design Rationale:
  - CNN extracts LOCAL artifacts (blending edges, texture, compression)
  - Transformer models GLOBAL spatial and TEMPORAL relationships

For single images:
  Spatial patches are extracted from CNN feature maps and fed as
  tokens to the Transformer encoder.

For video sequences:
  Each frame is processed by the CNN to produce frame-level features.
  These are assembled as a temporal sequence fed to the Transformer,
  which models temporal consistency (or inconsistency) across frames.

Architecture:

  [IMAGE MODE]
  Input (B, C, H, W)
       ↓
  CNN Feature Extractor              ← Shared EfficientNet-B4 backbone
       ↓
  Spatial Feature Map (B, C', H', W')
       ↓
  Reshape to sequence (B, N_tokens, C')  ← Each spatial location = token
       ↓
  Linear projection to embed_dim
       ↓
  [CLS] + Positional Encoding
       ↓
  Transformer Encoder × L layers
       ↓
  CLS Token → Classification Head
       ↓
  Real / Fake

  [VIDEO MODE]
  Input (B, T, C, H, W)             T = number of frames
       ↓
  CNN per frame → (B, T, D)         Temporal feature sequence
       ↓
  Positional Encoding (temporal)
       ↓
  Transformer Encoder × L layers
       ↓
  Mean Pooling → Classification Head
       ↓
  Real / Fake

Usage:
    from models.hybrid.hybrid_model import HybridModel
    model = HybridModel(mode='image')
    model_video = HybridModel(mode='video', n_frames=32)
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


# ── Positional Encoding (sinusoidal) ─────────────────────────────────────────
class SinusoidalPositionEncoding(nn.Module):
    """
    Fixed sinusoidal position encoding (Vaswani et al., 2017).
    Handles variable sequence lengths.
    """

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position encoding. x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── CNN Feature Extractor (shared backbone) ───────────────────────────────────
class CNNFeatureExtractor(nn.Module):
    """
    CNN backbone that can return:
      - Global pooled features (vector per image)
      - Spatial feature maps (for spatial tokenization)

    Args:
        backbone_name: timm backbone identifier.
        pretrained:    Use pretrained weights.
        output_dim:    Project features to this dimension.
        return_maps:   If True, return spatial maps (not pooled).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        pretrained: bool = True,
        output_dim: int = 512,
        return_maps: bool = False,
    ) -> None:
        super().__init__()
        self.return_maps = return_maps

        if TIMM_AVAILABLE:
            if return_maps:
                # Get feature maps before global pooling
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="",   # No pooling → returns feature maps
                )
                # Probe feature map channels
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    fmap = self.backbone(dummy)
                    C, H, W = fmap.shape[1], fmap.shape[2], fmap.shape[3]
                self.map_channels = C
                self.map_h = H
                self.map_w = W
                in_dim = C
            else:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224)
                    out = self.backbone(dummy)
                    in_dim = out.shape[1]
        else:
            # torchvision fallback
            from torchvision import models as tv_models
            m = tv_models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            if return_maps:
                self.backbone = m.features
                in_dim = 320
            else:
                m.classifier = nn.Identity()
                self.backbone = m
                in_dim = 1280

        # Projection to unified dimension
        self.proj = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            If return_maps=False: (B, output_dim) global features
            If return_maps=True:  (B, H*W, output_dim) spatial tokens
        """
        feat = self.backbone(x)

        if self.return_maps:
            B, C, H, W = feat.shape
            feat = feat.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)

        return self.proj(feat)


# ── Hybrid Model ──────────────────────────────────────────────────────────────
class HybridModel(nn.Module):
    """
    Hybrid CNN–Transformer model for deepfake detection.

    Modes:
      'image':  Single-image classification via spatial tokenization.
      'video':  Video classification via temporal frame sequence.

    Args:
        mode:           'image' | 'video'.
        backbone:       CNN backbone name (timm).
        pretrained:     Pretrained CNN backbone.
        num_classes:    Output classes.
        embed_dim:      Transformer embedding dimension.
        num_heads:      Attention heads.
        num_layers:     Transformer encoder layers.
        mlp_ratio:      MLP expansion ratio.
        dropout:        Dropout rate.
        n_frames:       Frames per video (video mode only).
        freeze_cnn:     Whether to freeze CNN weights during initial training.
    """

    def __init__(
        self,
        mode: str = "image",
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 2,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        n_frames: int = 32,
        freeze_cnn: bool = False,
    ) -> None:
        super().__init__()

        assert mode in ("image", "video"), f"mode must be 'image' or 'video'"
        self.mode = mode
        self.embed_dim = embed_dim
        self.n_frames = n_frames

        # ── CNN Feature Extractor ──────────────────────────────────────────────
        # For image: return spatial maps as tokens
        # For video: return global pooled features per frame
        self.cnn = CNNFeatureExtractor(
            backbone_name=backbone,
            pretrained=pretrained,
            output_dim=embed_dim,
            return_maps=(mode == "image"),
        )

        if freeze_cnn:
            for p in self.cnn.backbone.parameters():
                p.requires_grad = False

        # ── Learnable CLS token ───────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding ───────────────────────────────────────────────
        max_len = (14 * 14 + 1) if mode == "image" else (n_frames + 1)
        self.pos_enc = SinusoidalPositionEncoding(embed_dim, max_len=max_len + 16, dropout=dropout)

        # ── Transformer Encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image mode: (B, C, H, W)
               Video mode: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        if self.mode == "image":
            return self._forward_image(x)
        else:
            return self._forward_video(x)

    def _forward_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spatial tokenization: CNN feature map → Transformer → CLS logit.
        """
        B = x.shape[0]
        tokens = self.cnn(x)          # (B, n_tokens, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_tokens+1, D)

        # Add position encoding
        tokens = self.pos_enc(tokens)

        # Transformer
        out = self.transformer(tokens)  # (B, n_tokens+1, D)

        # Use CLS token for classification
        cls_out = out[:, 0]             # (B, D)
        return self.classifier(cls_out)

    def _forward_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal tokenization: each frame → CNN → token → Transformer → classify.
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # Process all frames through CNN (batched for efficiency)
        frames = x.view(B * T, C, H, W)
        frame_feats = self.cnn(frames)   # (B*T, D)
        frame_feats = frame_feats.view(B, T, -1)  # (B, T, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame_feats], dim=1)  # (B, T+1, D)

        # Positional encoding
        seq = self.pos_enc(seq)

        # Temporal transformer
        out = self.transformer(seq)    # (B, T+1, D)

        # Use CLS token for final prediction
        cls_out = out[:, 0]
        return self.classifier(cls_out)

    def get_frame_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        For video: return per-frame fake probability scores.
        Useful for suspicious frame identification.

        Args:
            x: (B, T, C, H, W)
        Returns:
            frame_probs: (B, T) — fake probability per frame
        """
        assert self.mode == "video"
        B, T, C, H, W = x.shape

        frames = x.view(B * T, C, H, W)
        frame_feats = self.cnn(frames)
        frame_feats = frame_feats.view(B, T, -1)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, frame_feats], dim=1)
        seq = self.pos_enc(seq)
        out = self.transformer(seq)

        # Score each frame position (not just CLS)
        frame_out = out[:, 1:]  # (B, T, D) — skip CLS
        frame_logits = self.classifier(frame_out)  # (B, T, 2)
        frame_probs = F.softmax(frame_logits, dim=-1)[:, :, 1]  # (B, T) fake prob
        return frame_probs

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_class, probabilities)."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "cnn": cnn_params,
            "transformer": transformer_params,
        }


# ── Factory ────────────────────────────────────────────────────────────────────
def build_hybrid_model(config, mode: str = "image") -> HybridModel:
    """Build HybridModel from config."""
    return HybridModel(
        mode=mode,
        backbone=config.model.cnn_backbone,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        embed_dim=config.model.transformer.embed_dim,
        num_heads=config.model.transformer.num_heads,
        num_layers=config.model.transformer.num_layers,
        mlp_ratio=config.model.transformer.mlp_ratio,
        dropout=config.model.transformer.dropout,
        n_frames=config.data.frames_per_video,
    )


# ── Quick sanity check ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing HybridModel [image mode]...")
    model_img = HybridModel(
        mode="image", pretrained=False,
        embed_dim=256, num_heads=4, num_layers=2,
    )
    x = torch.randn(2, 3, 224, 224)
    logits = model_img(x)
    params = model_img.count_parameters()
    print(f"  Image logits: {logits.shape}")
    print(f"  Params: {params['total']:,} (CNN={params['cnn']:,}, Transformer={params['transformer']:,})")

    print("\nTesting HybridModel [video mode]...")
    model_vid = HybridModel(
        mode="video", pretrained=False,
        embed_dim=256, num_heads=4, num_layers=2, n_frames=8,
    )
    xv = torch.randn(2, 8, 3, 224, 224)
    logits_v = model_vid(xv)
    frame_scores = model_vid.get_frame_scores(xv)
    print(f"  Video logits:      {logits_v.shape}")
    print(f"  Per-frame scores:  {frame_scores.shape}")
    print("HybridModel OK!")
