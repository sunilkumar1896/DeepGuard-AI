"""Sanity check all three model architectures."""
import sys
sys.path.insert(0, ".")

import torch
print("PyTorch:", torch.__version__)

# ── CNN ───────────────────────────────────────────────────────────────────────
print("\nTesting CNN model...")
from models.cnn.cnn_model import CNNModel
model = CNNModel(backbone="efficientnet_b0", pretrained=False, num_classes=2)
x = torch.randn(2, 3, 224, 224)
out = model(x)
params = model.count_parameters()
total = params["total"]
print(f"  CNN output: {out.shape}  params={total:,}")

# ── ViT ───────────────────────────────────────────────────────────────────────
print("\nTesting ViT model...")
from models.transformer.vit_model import ViTModel
vit = ViTModel(pretrained=False, embed_dim=128, num_heads=4, num_layers=2)
out2 = vit(x)
params2 = vit.count_parameters()
total2 = params2["total"]
mode2 = params2["mode"]
print(f"  ViT output: {out2.shape}  params={total2:,}  mode={mode2}")

# ── Hybrid image ──────────────────────────────────────────────────────────────
print("\nTesting Hybrid model [image]...")
from models.hybrid.hybrid_model import HybridModel
hybrid = HybridModel(mode="image", pretrained=False, embed_dim=128, num_heads=4, num_layers=2)
out3 = hybrid(x)
params3 = hybrid.count_parameters()
total3 = params3["total"]
cnn3 = params3["cnn"]
tr3 = params3["transformer"]
print(f"  Hybrid image output: {out3.shape}  params={total3:,}  CNN={cnn3:,}  Transformer={tr3:,}")

# ── Hybrid video ──────────────────────────────────────────────────────────────
print("\nTesting Hybrid model [video]...")
hybrid_v = HybridModel(mode="video", pretrained=False, embed_dim=128, num_heads=4, num_layers=2, n_frames=4)
xv = torch.randn(2, 4, 3, 224, 224)
out4 = hybrid_v(xv)
frame_scores = hybrid_v.get_frame_scores(xv)
print(f"  Hybrid video output:     {out4.shape}")
print(f"  Per-frame fake scores:   {frame_scores.shape}")

print("\n===========================")
print("ALL MODEL TESTS PASSED!")
print("===========================")
