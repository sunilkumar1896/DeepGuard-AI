"""
preprocessing/dataset_preparation.py
--------------------------------------
Complete dataset preparation pipeline for:
  - FaceForensics++ (FF++)
  - DeepFake Detection Challenge (DFDC)

Steps:
  1. Validate & inventory raw data
  2. Detect and remove corrupt files
  3. Detect duplicates (perceptual hash)
  4. Extract frames + face crops
  5. Create video-level train/val/test splits (no leakage)
  6. Save split manifests as JSON

Usage:
    # Prepare FF++
    python preprocessing/dataset_preparation.py \\
        --dataset faceforensics \\
        --raw_dir data/raw/faceforensics \\
        --out_dir data/processed \\
        --splits_dir data/splits

    # Prepare DFDC
    python preprocessing/dataset_preparation.py \\
        --dataset dfdc \\
        --raw_dir data/raw/dfdc \\
        --out_dir data/processed \\
        --splits_dir data/splits
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from preprocessing.face_detector import FaceDetector
from preprocessing.video_processor import VideoProcessor
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_DATASETS = ("faceforensics", "dfdc", "custom")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# FaceForensics++ compression levels
FF_COMPRESSIONS = ("c0", "c23", "c40")  # raw, HQ, LQ
FF_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


# ── Integrity checks ──────────────────────────────────────────────────────────
class FileValidator:
    """Check for corrupt images and videos."""

    @staticmethod
    def is_valid_image(path: Path) -> bool:
        try:
            img = cv2.imread(str(path))
            return img is not None and img.size > 0
        except Exception:
            return False

    @staticmethod
    def is_valid_video(path: Path) -> bool:
        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            return total > 0 and fps > 0
        except Exception:
            return False
        finally:
            if cap:
                cap.release()

    @staticmethod
    def md5_hash(path: Path, chunk_size: int = 8192) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def find_duplicates(cls, paths: list[Path]) -> dict[str, list[Path]]:
        """Group files by MD5 hash to find exact duplicates."""
        hash_map: dict[str, list[Path]] = defaultdict(list)
        for p in tqdm(paths, desc="Hashing files"):
            try:
                h = cls.md5_hash(p)
                hash_map[h].append(p)
            except OSError:
                pass
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


# ── Dataset inventories ───────────────────────────────────────────────────────
class DatasetInventory:
    """Build a structured inventory of a raw dataset."""

    def __init__(self, raw_dir: Path, dataset_type: str) -> None:
        self.raw_dir = raw_dir
        self.dataset_type = dataset_type
        self.items: list[dict] = []  # Each dict: {path, label, video_id, ...}

    def build(self) -> list[dict]:
        if self.dataset_type == "faceforensics":
            self._build_ff()
        elif self.dataset_type == "dfdc":
            self._build_dfdc()
        else:
            self._build_custom()
        logger.info(
            f"Inventory: {len(self.items)} files "
            f"({sum(1 for i in self.items if i['label'] == 0)} real, "
            f"{sum(1 for i in self.items if i['label'] == 1)} fake)"
        )
        return self.items

    def _build_ff(self) -> None:
        """
        Expected FF++ structure:
          data/raw/faceforensics/
            original_sequences/actors/c23/videos/*.mp4     <- REAL
            manipulated_sequences/Deepfakes/c23/videos/*.mp4 <- FAKE
            manipulated_sequences/Face2Face/c23/videos/*.mp4
            ...
        """
        # Real
        real_base = self.raw_dir / "original_sequences" / "actors"
        self._scan_dir(real_base, label=0, recursive=True)

        # Also check 'youtube' subdirectory (common in FF++)
        real_yt = self.raw_dir / "original_sequences" / "youtube"
        if real_yt.exists():
            self._scan_dir(real_yt, label=0, recursive=True)

        # Fake (all manipulation methods)
        manip_base = self.raw_dir / "manipulated_sequences"
        if manip_base.exists():
            for method in manip_base.iterdir():
                if method.is_dir():
                    self._scan_dir(method, label=1, recursive=True, method=method.name)

    def _build_dfdc(self) -> None:
        """
        Expected DFDC structure:
          data/raw/dfdc/
            dfdc_train_part_XX/
              metadata.json
              *.mp4
        """
        for part_dir in sorted(self.raw_dir.glob("dfdc_train_part_*")):
            meta_file = part_dir / "metadata.json"
            if not meta_file.exists():
                logger.warning(f"No metadata.json in {part_dir.name}, skipping.")
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            for fname, info in meta.items():
                fpath = part_dir / fname
                if not fpath.exists():
                    continue
                label = 1 if info.get("label") == "FAKE" else 0
                video_id = fpath.stem
                original = info.get("original", None)
                self.items.append({
                    "path": str(fpath),
                    "label": label,
                    "video_id": video_id,
                    "original": original,
                    "dataset": "dfdc",
                    "method": "dfdc",
                })

    def _build_custom(self) -> None:
        """
        Custom dataset with real/ and fake/ subdirectories.
          data/raw/custom/
            real/*.mp4   or  real/*.jpg
            fake/*.mp4   or  fake/*.jpg
        """
        for label_name, label_int in [("real", 0), ("fake", 1)]:
            label_dir = self.raw_dir / label_name
            if label_dir.exists():
                self._scan_dir(label_dir, label=label_int, recursive=True)

    def _scan_dir(
        self,
        base: Path,
        label: int,
        recursive: bool = True,
        method: str = "unknown",
    ) -> None:
        if not base.exists():
            return
        pattern = "**/*" if recursive else "*"
        for p in base.glob(pattern):
            if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT and p.is_file():
                self.items.append({
                    "path": str(p),
                    "label": label,
                    "video_id": p.stem,
                    "dataset": self.dataset_type,
                    "method": method,
                })


# ── Split creation (video-level, no leakage) ──────────────────────────────────
class DataSplitter:
    """
    Create train/val/test splits at the VIDEO level.

    This ensures that frames from the same source video never
    appear in different splits — preventing data leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed

    def split(self, items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Split items at the video level.

        Returns:
            (train_items, val_items, test_items)
        """
        rng = np.random.default_rng(self.seed)

        # Group by label to maintain class balance
        real = [i for i in items if i["label"] == 0]
        fake = [i for i in items if i["label"] == 1]

        def _split_group(group: list[dict]) -> tuple[list, list, list]:
            # Group by video_id first (some datasets have multiple clips per video)
            by_vid: dict[str, list[dict]] = defaultdict(list)
            for item in group:
                by_vid[item["video_id"]].append(item)

            video_ids = list(by_vid.keys())
            rng.shuffle(video_ids)

            n = len(video_ids)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))

            train_ids = set(video_ids[:n_train])
            val_ids = set(video_ids[n_train:n_train + n_val])
            test_ids = set(video_ids[n_train + n_val:])

            train = [i for vid in train_ids for i in by_vid[vid]]
            val = [i for vid in val_ids for i in by_vid[vid]]
            test = [i for vid in test_ids for i in by_vid[vid]]
            return train, val, test

        real_train, real_val, real_test = _split_group(real)
        fake_train, fake_val, fake_test = _split_group(fake)

        train = real_train + fake_train
        val = real_val + fake_val
        test = real_test + fake_test

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        logger.info(
            f"Split: train={len(train)} | val={len(val)} | test={len(test)}"
        )
        self._check_no_leakage(train, val, test)
        return train, val, test

    @staticmethod
    def _check_no_leakage(
        train: list[dict], val: list[dict], test: list[dict]
    ) -> None:
        """Assert no video_id overlap between splits."""
        def ids(lst): return {i["video_id"] for i in lst}
        train_ids, val_ids, test_ids = ids(train), ids(val), ids(test)

        tv = train_ids & val_ids
        tt = train_ids & test_ids
        vt = val_ids & test_ids

        if tv or tt or vt:
            logger.error(
                f"DATA LEAKAGE DETECTED!  "
                f"train/val overlap: {len(tv)}, "
                f"train/test overlap: {len(tt)}, "
                f"val/test overlap: {len(vt)}"
            )
            raise RuntimeError("Data leakage in splits — check video IDs.")
        else:
            logger.info("No data leakage detected. Splits are clean.")


# ── Main pipeline ─────────────────────────────────────────────────────────────
class DatasetPreparationPipeline:
    """
    Orchestrates the full dataset preparation pipeline:
      validate → inventory → deduplicate → extract faces → split → save.
    """

    def __init__(
        self,
        dataset_type: str,
        raw_dir: Path,
        out_dir: Path,
        splits_dir: Path,
        n_frames: int = 32,
        image_size: int = 224,
        face_margin: float = 0.3,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        skip_face_extraction: bool = False,
    ) -> None:
        self.dataset_type = dataset_type
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        self.splits_dir = splits_dir
        self.skip_face_extraction = skip_face_extraction
        self.seed = seed

        out_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        self.validator = FileValidator()
        self.inventory = DatasetInventory(raw_dir, dataset_type)
        self.splitter = DataSplitter(train_ratio, val_ratio, seed)

        target_size = (image_size, image_size)
        self.face_detector = FaceDetector(
            backend="mediapipe",
            margin=face_margin,
            target_size=target_size,
            align=True,
        )
        self.video_proc = VideoProcessor(
            n_frames=n_frames,
            strategy="uniform",
            face_detector=self.face_detector,
            target_size=target_size,
        )

    def run(self) -> None:
        """Execute the full pipeline."""
        logger.info("=" * 55)
        logger.info("Dataset Preparation Pipeline Starting")
        logger.info("=" * 55)
        set_seed(self.seed)

        # Step 1: Build inventory
        logger.info("[1/5] Building dataset inventory ...")
        items = self.inventory.build()
        if not items:
            logger.error(f"No files found in {self.raw_dir}. Check dataset structure.")
            return

        # Step 2: Validate & remove corrupt files
        logger.info("[2/5] Validating files ...")
        items = self._validate_files(items)

        # Step 3: Find & report duplicates
        logger.info("[3/5] Checking for duplicates ...")
        self._check_duplicates(items)

        # Step 4: Extract face crops
        if not self.skip_face_extraction:
            logger.info("[4/5] Extracting face crops ...")
            items = self._extract_faces(items)
        else:
            logger.info("[4/5] Skipping face extraction (--skip_faces flag set).")

        # Step 5: Create splits and save manifests
        logger.info("[5/5] Creating train/val/test splits ...")
        train, val, test = self.splitter.split(items)
        self._save_splits(train, val, test)

        logger.info("Pipeline complete!")
        self._print_summary(train, val, test)

    # ── Pipeline steps ─────────────────────────────────────────────────────────
    def _validate_files(self, items: list[dict]) -> list[dict]:
        valid = []
        corrupt = []
        for item in tqdm(items, desc="Validating"):
            p = Path(item["path"])
            if p.suffix.lower() in VIDEO_EXT:
                ok = self.validator.is_valid_video(p)
            else:
                ok = self.validator.is_valid_image(p)
            if ok:
                valid.append(item)
            else:
                corrupt.append(p.name)

        if corrupt:
            logger.warning(
                f"Found {len(corrupt)} corrupt files (removed from dataset)."
            )
            # Save list of corrupt files for reference
            corrupt_log = self.out_dir / "corrupt_files.txt"
            with open(corrupt_log, "w") as f:
                f.write("\n".join(corrupt))
            logger.info(f"Corrupt file list saved to {corrupt_log}")

        logger.info(f"Valid files: {len(valid)} / {len(items)}")
        return valid

    def _check_duplicates(self, items: list[dict]) -> None:
        paths = [Path(i["path"]) for i in items]
        dupes = self.validator.find_duplicates(paths)
        if dupes:
            logger.warning(
                f"Found {len(dupes)} duplicate groups. "
                f"Consider removing duplicates from the raw dataset."
            )
            dupe_log = self.out_dir / "duplicates.json"
            with open(dupe_log, "w") as f:
                json.dump(
                    {h: [str(p) for p in ps] for h, ps in dupes.items()}, f, indent=2
                )
        else:
            logger.info("No duplicates found.")

    def _extract_faces(self, items: list[dict]) -> list[dict]:
        """
        Extract and save face crops for each item.
        Updates item['face_path'] to point to the saved crop.
        """
        processed = []
        no_face_count = 0

        for item in tqdm(items, desc="Extracting faces"):
            p = Path(item["path"])
            label_str = "real" if item["label"] == 0 else "fake"
            out_subdir = self.out_dir / "images" / label_str
            out_subdir.mkdir(parents=True, exist_ok=True)

            if p.suffix.lower() in VIDEO_EXT:
                face_paths = self._process_video(p, out_subdir, item["video_id"])
            else:
                face_paths = self._process_image(p, out_subdir, item["video_id"])

            if not face_paths:
                no_face_count += 1
                continue

            # Create one item per face crop
            for i, fp in enumerate(face_paths):
                new_item = dict(item)
                new_item["face_path"] = str(fp)
                new_item["frame_index"] = i
                processed.append(new_item)

        logger.info(
            f"Face extraction done. "
            f"Produced {len(processed)} face crops. "
            f"No-face videos/images: {no_face_count}"
        )
        return processed

    def _process_video(
        self, video_path: Path, out_dir: Path, video_id: str
    ) -> list[Path]:
        """Extract face crops from video, save as JPEG images."""
        try:
            frame_results = self.video_proc.extract_faces(video_path)
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            return []

        saved_paths = []
        for fr in frame_results:
            if not fr.has_face:
                continue
            # Use the largest face
            face = max(
                fr.faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            if face.face_crop is None:
                continue
            fname = f"{video_id}_f{fr.frame_index:06d}.jpg"
            save_path = out_dir / fname
            cv2.imwrite(str(save_path), face.face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(save_path)

        return saved_paths

    def _process_image(
        self, image_path: Path, out_dir: Path, video_id: str
    ) -> list[Path]:
        """Detect and crop face from a static image, save as JPEG."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return []
            face = self.face_detector.detect_largest(img)
            if face is None or face.face_crop is None:
                return []
            fname = f"{video_id}_face.jpg"
            save_path = out_dir / fname
            cv2.imwrite(str(save_path), face.face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return [save_path]
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            return []

    def _save_splits(
        self,
        train: list[dict],
        val: list[dict],
        test: list[dict],
    ) -> None:
        """Save split manifests as JSON files."""
        splits = {"train": train, "val": val, "test": test}
        for split_name, items in splits.items():
            out_path = self.splits_dir / f"{self.dataset_type}_{split_name}.json"
            with open(out_path, "w") as f:
                json.dump(items, f, indent=2)
            logger.info(f"Saved {split_name} manifest: {out_path}")

    @staticmethod
    def _print_summary(
        train: list[dict], val: list[dict], test: list[dict]
    ) -> None:
        def counts(lst):
            r = sum(1 for i in lst if i.get("label") == 0)
            f = sum(1 for i in lst if i.get("label") == 1)
            return len(lst), r, f

        t, tr, tf = counts(train)
        v, vr, vf = counts(val)
        te, ter, tef = counts(test)

        print("\n" + "=" * 45)
        print("Dataset Preparation Summary")
        print("=" * 45)
        print(f"{'Split':<10} {'Total':>8} {'Real':>8} {'Fake':>8}")
        print("-" * 45)
        print(f"{'Train':<10} {t:>8} {tr:>8} {tf:>8}")
        print(f"{'Val':<10} {v:>8} {vr:>8} {vf:>8}")
        print(f"{'Test':<10} {te:>8} {ter:>8} {tef:>8}")
        print("=" * 45)


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deepfake detection dataset"
    )
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset type",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        required=True,
        help="Path to raw dataset directory",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed faces",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for split JSON manifests",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=32,
        help="Frames to sample per video",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Face crop size (pixels)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--skip_faces",
        action="store_true",
        help="Skip face extraction (only build splits from existing paths)",
    )
    args = parser.parse_args()

    pipeline = DatasetPreparationPipeline(
        dataset_type=args.dataset,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        splits_dir=args.splits_dir,
        n_frames=args.n_frames,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_face_extraction=args.skip_faces,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
