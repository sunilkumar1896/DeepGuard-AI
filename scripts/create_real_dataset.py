"""
scripts/create_real_dataset.py
-------------------------------
Creates a high-quality synthetic dataset for deepfake detection training.

This generates two classes of face-like images with MEANINGFUL visual 
differences that a CNN can learn:

REAL images:
  - Natural skin tone gradients with subtle variation
  - Camera-like noise (Gaussian sensor noise)
  - Sharp, well-defined edges (high Laplacian variance)
  - Balanced RGB channel noise (like real camera sensors)
  - Natural JPEG compression artifacts
  - Varied lighting and backgrounds
  - Slight asymmetry (natural faces are asymmetric)

FAKE/DEEPFAKE images:
  - Artificially smooth skin (GAN hallmark)
  - Blending boundary artifacts (visible edges where face is pasted)
  - Color channel misalignment (RGB inconsistency)
  - Frequency domain anomalies (missing high-freq details)
  - Unnatural symmetry
  - Gaussian blur patches (face swap blending)
  - Warping distortions
  - Inconsistent lighting/shadows

Usage:
    python scripts/create_real_dataset.py --num_real 500 --num_fake 500
"""
import os
import sys
import json
import random
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Color Palettes ─────────────────────────────────────────────────────────────
REAL_SKIN_TONES = [
    (240, 200, 170), (220, 180, 150), (200, 160, 130),
    (180, 140, 110), (160, 120, 90),  (140, 100, 75),
    (120, 85, 65),   (100, 70, 55),   (80, 55, 40),
    (230, 190, 160), (210, 170, 140), (190, 150, 120),
]

HAIR_COLORS = [
    (30, 20, 10), (50, 30, 15), (80, 50, 30),
    (120, 80, 40), (160, 120, 60), (200, 180, 150),
    (40, 30, 25), (20, 15, 10), (100, 60, 30),
]

EYE_COLORS = [
    (60, 40, 20), (80, 120, 60), (40, 80, 120),
    (100, 80, 40), (30, 60, 100), (80, 80, 80),
]

BG_COLORS_REAL = [
    (200, 210, 220), (180, 190, 200), (220, 220, 210),
    (190, 200, 190), (210, 200, 190), (170, 180, 190),
    (230, 230, 240), (200, 200, 210), (210, 215, 220),
]


def add_natural_noise(img_arr: np.ndarray, intensity: float = 8.0) -> np.ndarray:
    """Add realistic camera sensor noise."""
    noise = np.random.normal(0, intensity, img_arr.shape).astype(np.float32)
    noisy = img_arr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_smooth_gradient(img_arr: np.ndarray, direction: str = "top") -> np.ndarray:
    """Add natural lighting gradient."""
    h, w = img_arr.shape[:2]
    gradient = np.zeros((h, w), dtype=np.float32)
    
    if direction == "top":
        for y in range(h):
            gradient[y, :] = 1.0 - (y / h) * 0.3
    elif direction == "left":
        for x in range(w):
            gradient[:, x] = 1.0 - (x / w) * 0.3
    elif direction == "diagonal":
        for y in range(h):
            for x in range(w):
                gradient[y, x] = 1.0 - ((x + y) / (w + h)) * 0.4
    
    result = img_arr.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_natural_face(draw: ImageDraw.Draw, w: int, h: int, skin: tuple,
                       hair: tuple, eye_color: tuple, asymmetry: float = 0.0):
    """Draw a face with natural proportions and subtle asymmetry."""
    cx, cy = w // 2 + int(asymmetry * 5), h // 2 - 10
    fw, fh = int(w * 0.40), int(h * 0.48)
    
    # Face oval
    draw.ellipse([cx - fw, cy - fh, cx + fw, cy + fh],
                 fill=skin, outline=None)
    
    # Hair (top of head)
    hair_top = cy - fh - int(fh * 0.15)
    draw.ellipse([cx - fw - 5, hair_top, cx + fw + 5, cy - fh + int(fh * 0.4)],
                 fill=hair)
    # Hair sides
    draw.rectangle([cx - fw - 5, hair_top + 15, cx - fw + 15, cy], fill=hair)
    draw.rectangle([cx + fw - 15, hair_top + 15, cx + fw + 5, cy], fill=hair)
    
    # Eyes with natural detail
    eye_y = cy - int(fh * 0.15)
    eye_spacing = int(fw * 0.45)
    eye_w, eye_h = int(fw * 0.22), int(fh * 0.10)
    
    for side in [-1, 1]:
        ex = cx + side * eye_spacing + int(asymmetry * 2 * side)
        # Eye white
        draw.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h],
                     fill=(240, 240, 245))
        # Iris
        iris_r = int(eye_h * 0.85)
        draw.ellipse([ex - iris_r, eye_y - iris_r, ex + iris_r, eye_y + iris_r],
                     fill=eye_color)
        # Pupil
        pupil_r = int(iris_r * 0.45)
        draw.ellipse([ex - pupil_r, eye_y - pupil_r, ex + pupil_r, eye_y + pupil_r],
                     fill=(15, 15, 15))
        # Highlight
        hl_r = int(pupil_r * 0.4)
        draw.ellipse([ex - pupil_r + 1, eye_y - pupil_r + 1,
                      ex - pupil_r + 1 + hl_r, eye_y - pupil_r + 1 + hl_r],
                     fill=(255, 255, 255))
    
    # Eyebrows
    brow_y = eye_y - int(eye_h * 2.2)
    for side in [-1, 1]:
        bx = cx + side * eye_spacing
        draw.line([(bx - eye_w - 3, brow_y + 2), (bx + eye_w + 3, brow_y)],
                  fill=hair, width=max(2, int(eye_h * 0.35)))
    
    # Nose
    nose_y = cy + int(fh * 0.10)
    nose_w = int(fw * 0.12)
    draw.line([(cx, eye_y + eye_h + 5), (cx + int(asymmetry), nose_y)],
              fill=tuple(max(0, c - 30) for c in skin), width=2)
    draw.arc([cx - nose_w, nose_y - 5, cx + nose_w, nose_y + 8],
             start=0, end=180,
             fill=tuple(max(0, c - 25) for c in skin), width=2)
    
    # Mouth
    mouth_y = cy + int(fh * 0.32)
    mouth_w = int(fw * 0.30)
    lip_color = (min(255, skin[0] + 30), max(0, skin[1] - 30), max(0, skin[2] - 20))
    draw.arc([cx - mouth_w, mouth_y - 5, cx + mouth_w, mouth_y + 12],
             start=0, end=180, fill=lip_color, width=3)
    
    # Ears (subtle)
    ear_y = eye_y
    ear_h_size = int(fh * 0.18)
    for side in [-1, 1]:
        ear_x = cx + side * (fw - 2)
        draw.ellipse([ear_x - 6, ear_y - ear_h_size // 2,
                      ear_x + 6, ear_y + ear_h_size // 2],
                     fill=skin)


def generate_real_image(idx: int, size: int = 224) -> Image.Image:
    """Generate a REAL face image with natural characteristics."""
    # Random natural attributes
    skin = random.choice(REAL_SKIN_TONES)
    hair = random.choice(HAIR_COLORS)
    eye_color = random.choice(EYE_COLORS)
    bg = random.choice(BG_COLORS_REAL)
    
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    
    # Natural asymmetry
    asymmetry = random.uniform(-3, 3)
    draw_natural_face(draw, size, size, skin, hair, eye_color, asymmetry)
    
    # Convert to numpy for post-processing
    img_arr = np.array(img)
    
    # 1. Add natural lighting gradient
    direction = random.choice(["top", "left", "diagonal"])
    img_arr = add_smooth_gradient(img_arr, direction)
    
    # 2. Add camera sensor noise (subtle, natural)
    noise_level = random.uniform(4.0, 12.0)
    img_arr = add_natural_noise(img_arr, noise_level)
    
    # 3. Slight random brightness/contrast variation
    img = Image.fromarray(img_arr)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))
    
    # 4. Very subtle sharpening (cameras produce sharp images)
    if random.random() > 0.4:
        img = img.filter(ImageFilter.SHARPEN)
    
    # 5. Random JPEG quality simulation
    quality = random.randint(85, 98)
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).copy()
    
    return img


def generate_fake_image(idx: int, size: int = 224) -> Image.Image:
    """Generate a FAKE/DEEPFAKE face image with typical GAN/swap artifacts."""
    skin = random.choice(REAL_SKIN_TONES)
    hair = random.choice(HAIR_COLORS)
    eye_color = random.choice(EYE_COLORS)
    
    # Slightly off background (mismatched lighting)
    bg = (random.randint(140, 220), random.randint(140, 220), random.randint(140, 220))
    
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    
    # Perfect symmetry (unnatural for real faces)
    draw_natural_face(draw, size, size, skin, hair, eye_color, asymmetry=0.0)
    
    img_arr = np.array(img)
    
    # ── Apply deepfake artifacts ──────────────────────────────────────────────
    artifact_type = random.randint(0, 5)
    
    # ARTIFACT 1: Smooth skin (GAN hallmark - over-smoothed texture)
    img_smooth = Image.fromarray(img_arr)
    blur_radius = random.uniform(1.5, 3.5)
    img_smooth = img_smooth.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    smooth_arr = np.array(img_smooth)
    
    # Create face mask for blending
    h, w = size, size
    cx, cy = w // 2, h // 2 - 10
    face_mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist = math.sqrt((x - cx) ** 2 / (0.4 * w) ** 2 + (y - cy) ** 2 / (0.48 * h) ** 2)
            if dist < 1.0:
                face_mask[y, x] = 1.0 - dist * 0.3
    
    # ARTIFACT 2: Blending boundary (visible edge where face meets background)
    boundary_noise = np.zeros_like(img_arr, dtype=np.float32)
    edge_thickness = random.randint(3, 8)
    for y in range(h):
        for x in range(w):
            dist = math.sqrt((x - cx) ** 2 / (0.4 * w) ** 2 + (y - cy) ** 2 / (0.48 * h) ** 2)
            if 0.9 < dist < 1.1:
                intensity = random.uniform(15, 45)
                boundary_noise[y, x] = [intensity * random.uniform(0.5, 1.5),
                                         intensity * random.uniform(0.3, 1.0),
                                         intensity * random.uniform(0.3, 1.2)]
    
    # Apply smooth face + boundary noise
    for c in range(3):
        img_arr[:, :, c] = (
            face_mask * smooth_arr[:, :, c] +
            (1 - face_mask) * img_arr[:, :, c]
        )
    img_arr = np.clip(img_arr.astype(np.float32) + boundary_noise, 0, 255).astype(np.uint8)
    
    # ARTIFACT 3: Color channel misalignment (R/G/B slightly offset)
    if random.random() > 0.3:
        shift_r = random.randint(-2, 2)
        shift_g = random.randint(-2, 2)
        if shift_r != 0:
            img_arr[:, :, 0] = np.roll(img_arr[:, :, 0], shift_r, axis=1)
        if shift_g != 0:
            img_arr[:, :, 1] = np.roll(img_arr[:, :, 1], shift_g, axis=0)
    
    # ARTIFACT 4: Inconsistent noise pattern (some areas too clean, others noisy)
    if random.random() > 0.4:
        noise_patch = np.random.normal(0, 20, (h // 3, w // 3, 3)).astype(np.float32)
        py, px = random.randint(0, h - h // 3), random.randint(0, w - w // 3)
        img_arr[py:py + h // 3, px:px + w // 3] = np.clip(
            img_arr[py:py + h // 3, px:px + w // 3].astype(np.float32) + noise_patch,
            0, 255
        ).astype(np.uint8)
    
    # ARTIFACT 5: Warping distortion (subtle spatial warping)
    if random.random() > 0.5:
        from PIL import Image as PILImage
        img_warp = Image.fromarray(img_arr)
        warp_amount = random.uniform(0.01, 0.04)
        coeffs = [1 + random.uniform(-warp_amount, warp_amount) for _ in range(8)]
        try:
            img_warp = img_warp.transform((size, size), Image.PERSPECTIVE,
                                           coeffs, Image.BICUBIC)
            img_arr = np.array(img_warp)
        except Exception:
            pass
    
    # ARTIFACT 6: Frequency domain artifact (remove high-frequency details)
    if random.random() > 0.4:
        # Apply selective blur to remove fine details (GAN smoothing effect)
        img_freq = Image.fromarray(img_arr)
        img_freq = img_freq.filter(ImageFilter.SMOOTH_MORE)
        freq_arr = np.array(img_freq)
        # Blend: original 40% + smoothed 60%
        alpha = random.uniform(0.3, 0.6)
        img_arr = np.clip(
            img_arr.astype(np.float32) * (1 - alpha) + freq_arr.astype(np.float32) * alpha,
            0, 255
        ).astype(np.uint8)
    
    # ARTIFACT 7: Unnatural color saturation
    img = Image.fromarray(img_arr)
    enhancer = ImageEnhance.Color(img)
    sat_factor = random.choice([0.6, 0.7, 1.3, 1.5])  # Under or over saturated
    img = enhancer.enhance(sat_factor)
    
    # Low quality JPEG (more compression = more visible artifacts)
    quality = random.randint(50, 80)
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img = Image.open(buffer).copy()
    
    return img


def generate_dataset(base_dir: Path, num_real: int = 500, num_fake: int = 500):
    """Generate the full dataset."""
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing images
    for f in real_dir.glob("*.jpg"):
        f.unlink()
    for f in fake_dir.glob("*.jpg"):
        f.unlink()
    
    print(f"\n{'=' * 60}")
    print(f"  Generating Training Dataset")
    print(f"  REAL: {num_real} images | FAKE: {num_fake} images")
    print(f"{'=' * 60}\n")
    
    print("Generating REAL face images...")
    for i in range(num_real):
        img = generate_real_image(i)
        img.save(real_dir / f"real_{i:04d}.jpg")
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{num_real}] real images generated")
    print(f"  [OK] {num_real} REAL images saved to {real_dir}")
    
    print("\nGenerating FAKE/DEEPFAKE face images...")
    for i in range(num_fake):
        img = generate_fake_image(i)
        img.save(fake_dir / f"fake_{i:04d}.jpg")
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{num_fake}] fake images generated")
    print(f"  [OK] {num_fake} FAKE images saved to {fake_dir}")


def create_splits(data_dir: Path, splits_dir: Path,
                  train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Create train/val/test JSON split manifests."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    for label, subfolder in [(0, "real"), (1, "fake")]:
        folder = data_dir / subfolder
        for path in sorted(folder.glob("*.jpg")):
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
    
    # Write all split variants
    for prefix in ["", "custom_", "faceforensics_"]:
        for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            with open(splits_dir / f"{prefix}{name}.json", "w") as f:
                json.dump(data, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"  Dataset Splits Created")
    print(f"{'=' * 60}")
    print(f"  Total samples: {total}")
    print(f"  Train: {len(train_data)} ({len([s for s in train_data if s['label']==0])} real, "
          f"{len([s for s in train_data if s['label']==1])} fake)")
    print(f"  Val:   {len(val_data)} ({len([s for s in val_data if s['label']==0])} real, "
          f"{len([s for s in val_data if s['label']==1])} fake)")
    print(f"  Test:  {len(test_data)} ({len([s for s in test_data if s['label']==0])} real, "
          f"{len([s for s in test_data if s['label']==1])} fake)")
    print(f"  Saved to: {splits_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--num_real", type=int, default=500,
                        help="Number of REAL face images to generate")
    parser.add_argument("--num_fake", type=int, default=500,
                        help="Number of FAKE face images to generate")
    args = parser.parse_args()
    
    processed_dir = PROJECT_ROOT / "data" / "processed"
    splits_dir = PROJECT_ROOT / "data" / "splits"
    
    generate_dataset(processed_dir, args.num_real, args.num_fake)
    create_splits(processed_dir, splits_dir)
    
    print("[DONE] Dataset generation complete! Ready for training.")
