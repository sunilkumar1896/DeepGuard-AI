"""
scripts/download_deepfakeface.py
---------------------------------
Downloads the OpenRL/DeepFakeFace benchmark dataset directly from Hugging Face:
  - wiki.zip: Real human faces (IMDB-WIKI celebrity faces)
  - insight.zip: Fake faces (InsightFace facial deepfakes / face swaps)

Extracts images into data/processed/real and data/processed/fake,
rebuilds train/val/test splits, and packs dataset.zip.
"""
import os
import sys
import zipfile
import json
import random
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
PROCESSED_DIR = DATA_DIR / "processed"
REAL_DIR = PROCESSED_DIR / "real"
FAKE_DIR = PROCESSED_DIR / "fake"
SPLITS_DIR = DATA_DIR / "splits"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset_files(include_diffusion: bool = False):
    """Downloads zip files from OpenRL/DeepFakeFace on Hugging Face."""
    print("=" * 60)
    print("  Downloading OpenRL/DeepFakeFace from Hugging Face")
    print("=" * 60)

    files_to_download = [
        ("wiki.zip", "REAL human faces (IMDB-WIKI)"),
        ("insight.zip", "FAKE faces (InsightFace swaps)"),
    ]
    if include_diffusion:
        files_to_download.append(("text2img.zip", "FAKE faces (Stable Diffusion AI)"))

    downloaded = {}
    for filename, desc in files_to_download:
        dest_path = DOWNLOADS_DIR / filename
        if dest_path.exists() and dest_path.stat().st_size > 10 * 1024 * 1024:
            print(f"  [EXISTS] {filename} ({round(dest_path.stat().st_size / (1024*1024), 1)} MB)")
            downloaded[filename] = dest_path
            continue

        print(f"  [DOWNLOADING] {filename} — {desc}...")
        file_path = hf_hub_download(
            repo_id="OpenRL/DeepFakeFace",
            filename=filename,
            repo_type="dataset",
            local_dir=str(DOWNLOADS_DIR),
            local_dir_use_symlinks=False,
        )
        downloaded[filename] = Path(file_path)
        print(f"  [DONE] Saved to {file_path}")

    return downloaded


def extract_samples(zip_path: Path, output_dir: Path, prefix: str, max_samples: int = 5000):
    """Extracts image files from zip into target directory with clean names."""
    print(f"\nExtracting up to {max_samples} images from {zip_path.name} -> {output_dir.name}/...")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    with zipfile.ZipFile(zip_path, "r") as z:
        img_members = [
            m for m in z.namelist()
            if not m.startswith("__MACOSX") and Path(m).suffix.lower() in valid_exts
        ]
        print(f"  Total valid images in archive: {len(img_members):,}")

        selected = img_members[:max_samples] if max_samples else img_members
        extracted = 0

        for i, member in enumerate(selected):
            ext = Path(member).suffix.lower() or ".jpg"
            out_name = f"{prefix}_{i:05d}{ext}"
            out_path = output_dir / out_name

            with z.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            extracted += 1
            if extracted % 1000 == 0 or extracted == len(selected):
                print(f"    Extracted {extracted:,} / {len(selected):,} images...")

    return extracted


def build_split_manifests(train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Generates train.json, val.json, test.json split manifests."""
    print("\nGenerating split manifests in data/splits/...")
    samples = []

    for img_path in sorted(REAL_DIR.glob("*.*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rel_path = f"data/processed/real/{img_path.name}"
            samples.append({
                "image_path": rel_path,
                "face_path": rel_path,
                "label": 0,
                "label_name": "real",
            })

    for img_path in sorted(FAKE_DIR.glob("*.*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            rel_path = f"data/processed/fake/{img_path.name}"
            samples.append({
                "image_path": rel_path,
                "face_path": rel_path,
                "label": 1,
                "label_name": "fake",
            })

    random.seed(42)
    random.shuffle(samples)

    total = len(samples)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]

    for name, data in [
        ("train.json", train_data),
        ("val.json", val_data),
        ("test.json", test_data),
        ("custom_train.json", train_data),
        ("custom_val.json", val_data),
        ("custom_test.json", test_data),
        ("faceforensics_train.json", train_data),
        ("faceforensics_val.json", val_data),
        ("faceforensics_test.json", test_data),
    ]:
        with open(SPLITS_DIR / name, "w") as f:
            json.dump(data, f, indent=2)

    print(f"  [OK] Total samples: {total:,}")
    print(f"       Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")


def main():
    parser = argparse.ArgumentParser(description="Download and organize DeepFakeFace dataset")
    parser.add_argument("--max_per_class", type=int, default=5000, help="Max images per class to extract (default: 5000)")
    parser.add_argument("--include_diffusion", action="store_true", help="Also download text2img.zip diffusion faces")
    args = parser.parse_args()

    # 1. Download
    downloaded = download_dataset_files(include_diffusion=args.include_diffusion)

    # 2. Extract REAL
    extract_samples(downloaded["wiki.zip"], REAL_DIR, prefix="real", max_samples=args.max_per_class)

    # 3. Extract FAKE
    extract_samples(downloaded["insight.zip"], FAKE_DIR, prefix="fake", max_samples=args.max_per_class)

    # 4. Generate Splits
    build_split_manifests()

    print("\n" + "=" * 60)
    print("  Dataset integration complete!")
    print(f"  Real images: {len(list(REAL_DIR.glob('*.*'))):,}")
    print(f"  Fake images: {len(list(FAKE_DIR.glob('*.*'))):,}")
    print("=" * 60)


if __name__ == "__main__":
    main()

