"""
scripts/create_demo_dataset.py
------------------------------
Generates a structured local dataset of REAL and DEEPFAKE images/faces,
and creates the train/val/test JSON split manifests required for model training.

Usage:
    python scripts/create_demo_dataset.py --num_samples 100
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_sample_images(base_dir: Path, num_real: int = 50, num_fake: int = 50):
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_real} REAL and {num_fake} FAKE sample facial images...")

    # Generate REAL face samples (smooth gradients, natural skin tones, sharp eyes)
    for i in range(num_real):
        img = Image.new("RGB", (224, 224), color=(random.randint(180, 240), random.randint(140, 190), random.randint(120, 160)))
        draw = ImageDraw.Draw(img)
        # Face outline
        draw.ellipse([40, 30, 184, 194], fill=(225, 175, 145), outline=(180, 130, 100), width=2)
        # Eyes
        draw.ellipse([70, 80, 95, 100], fill=(255, 255, 255))
        draw.ellipse([80, 87, 88, 95], fill=(30, 30, 30))
        draw.ellipse([129, 80, 154, 100], fill=(255, 255, 255))
        draw.ellipse([139, 87, 147, 95], fill=(30, 30, 30))
        # Mouth
        draw.arc([85, 140, 139, 165], start=0, end=180, fill=(180, 60, 60), width=3)
        img.save(real_dir / f"real_{i:04d}.jpg")

    # Generate FAKE face samples (blending artifacts, noise distortion, color mismatch)
    for i in range(num_fake):
        img = Image.new("RGB", (224, 224), color=(random.randint(160, 220), random.randint(120, 180), random.randint(140, 200)))
        draw = ImageDraw.Draw(img)
        # Face outline
        draw.ellipse([40, 30, 184, 194], fill=(215, 160, 135), outline=(230, 80, 80), width=4)
        # Artificial Eyes
        draw.ellipse([68, 78, 97, 102], fill=(200, 220, 255))
        draw.ellipse([78, 85, 87, 95], fill=(80, 20, 20))
        draw.ellipse([127, 78, 156, 102], fill=(200, 220, 255))
        draw.ellipse([137, 85, 146, 95], fill=(80, 20, 20))
        # Distorted mouth blending boundary
        draw.rectangle([75, 135, 149, 170], fill=(190, 80, 90))
        # Apply synthetic blur/noise artifact
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        img.save(fake_dir / f"fake_{i:04d}.jpg")

    print("Dataset image generation complete.")


def create_splits(data_dir: Path, splits_dir: Path, train_ratio=0.7, val_ratio=0.15):
    splits_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for label, subfolder in [(0, "real"), (1, "fake")]:
        folder = data_dir / subfolder
        for path in folder.glob("*.jpg"):
            rel_path = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            all_samples.append({
                "image_path": rel_path,
                "face_path": rel_path,
                "label": label,
                "label_name": "real" if label == 0 else "fake"
            })

    random.seed(42)
    random.shuffle(all_samples)

    total = len(all_samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_data = all_samples[:n_train]
    val_data = all_samples[n_train:n_train + n_val]
    test_data = all_samples[n_train + n_val:]

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(splits_dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)
        with open(splits_dir / f"custom_{name}.json", "w") as f:
            json.dump(data, f, indent=2)
        with open(splits_dir / f"faceforensics_{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    print(f"Created JSON splits in {splits_dir}:")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Val samples:   {len(val_data)}")
    print(f"  - Test samples:  {len(test_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_real", type=int, default=50)
    parser.add_argument("--num_fake", type=int, default=50)
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed"
    splits_dir = PROJECT_ROOT / "data" / "splits"

    generate_sample_images(processed_dir, args.num_real, args.num_fake)
    create_splits(processed_dir, splits_dir)
