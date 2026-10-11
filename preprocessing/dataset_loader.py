"""
preprocessing/dataset_loader.py
---------------------------------
PyTorch Dataset classes for the deepfake detection platform.

Classes:
  - DeepfakeImageDataset: Single-image classification dataset.
  - DeepfakeVideoDataset: Video-level dataset returning frame sequences.
  - DeepfakeDataModule:   High-level wrapper that creates all DataLoaders.

The datasets read from JSON manifests produced by dataset_preparation.py.
Each JSON manifest entry has at minimum:
  { "face_path": "/path/to/face.jpg", "label": 0 or 1, "video_id": "..." }
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Image Dataset ──────────────────────────────────────────────────────────────
class DeepfakeImageDataset(Dataset):
    """
    Dataset of face-cropped images for binary deepfake classification.

    Args:
        manifest_path:  Path to JSON manifest file from dataset_preparation.py.
        transform:      Albumentations or torchvision transform pipeline.
        use_albumentations: Whether transform is an albumentations pipeline.
        label_key:      Key name for the label in manifest entries.
        face_key:       Key name for the face image path.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transform=None,
        use_albumentations: bool = True,
        label_key: str = "label",
        face_key: str = "face_path",
    ) -> None:
        self.transform = transform
        self.use_albumentations = use_albumentations
        self.label_key = label_key
        self.face_key = face_key

        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            self.items: list[dict] = json.load(f)

        # Filter out items missing face_path
        before = len(self.items)
        self.items = [i for i in self.items if face_key in i]
        if len(self.items) < before:
            logger.warning(
                f"Dropped {before - len(self.items)} items missing '{face_key}' key."
            )

        self._labels = [i[label_key] for i in self.items]
        logger.info(
            f"Loaded dataset: {len(self.items)} samples  "
            f"(real={self._labels.count(0)}, fake={self._labels.count(1)})"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.items[idx]
        img = self._load_image(item[self.face_key])

        if self.transform is not None:
            img = self._apply_transform(img)
        else:
            img = self._default_to_tensor(img)

        label = int(item[self.label_key])
        return img, label

    def get_labels(self) -> list[int]:
        return self._labels

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            # Return blank image if file missing (dataset integrity issue)
            logger.warning(f"Could not load image: {path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_transform(self, rgb: np.ndarray) -> torch.Tensor:
        if self.use_albumentations:
            try:
                result = self.transform(image=rgb)
                return result["image"]
            except TypeError:
                return self.transform(rgb)
        else:
            return self.transform(rgb)

    @staticmethod
    def _default_to_tensor(rgb: np.ndarray) -> torch.Tensor:
        """Convert to float32 tensor [C, H, W] without normalization."""
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        return tensor


# ── Video Dataset ──────────────────────────────────────────────────────────────
class DeepfakeVideoDataset(Dataset):
    """
    Video-level dataset that returns a sequence of frame tensors.
    Used for the temporal Transformer and Hybrid models.

    Returns a tensor of shape (T, C, H, W) where T = n_frames.
    If a video has fewer than n_frames, it is padded with the last frame.

    Args:
        manifest_path:  Path to JSON manifest.
        n_frames:       Sequence length for temporal model.
        transform:      Applied to each individual frame.
        use_albumentations: Whether transform is albumentations-based.
        video_id_key:   Key to group frames by video.
        label_key:      Key for label.
        face_key:       Key for face image path.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        n_frames: int = 32,
        transform=None,
        use_albumentations: bool = True,
        video_id_key: str = "video_id",
        label_key: str = "label",
        face_key: str = "face_path",
    ) -> None:
        self.n_frames = n_frames
        self.transform = transform
        self.use_albumentations = use_albumentations

        with open(Path(manifest_path), "r") as f:
            all_items: list[dict] = json.load(f)

        # Group frames by video_id
        from collections import defaultdict
        video_groups: dict[str, list[dict]] = defaultdict(list)
        for item in all_items:
            if face_key in item:
                video_groups[item[video_id_key]].append(item)

        # Build video-level list
        self.videos: list[dict] = []
        for vid_id, frames in video_groups.items():
            # All frames of same video should have same label
            label = frames[0][label_key]
            self.videos.append({
                "video_id": vid_id,
                "label": label,
                "frames": sorted(
                    frames,
                    key=lambda x: x.get("frame_index", 0)
                ),
            })

        self._labels = [v["label"] for v in self.videos]
        logger.info(
            f"VideoDataset: {len(self.videos)} videos  "
            f"(real={self._labels.count(0)}, fake={self._labels.count(1)})"
        )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video = self.videos[idx]
        frames = video["frames"]
        label = int(video["label"])

        # Sample or pad to n_frames
        frame_paths = [f["face_path"] for f in frames]
        frame_paths = self._sample_frames(frame_paths)

        # Load and transform each frame
        frame_tensors = []
        for fp in frame_paths:
            img = self._load_image(fp)
            if self.transform is not None:
                t = self._apply_transform(img)
            else:
                t = self._default_to_tensor(img)
            frame_tensors.append(t)

        # Stack to (T, C, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)
        return video_tensor, label

    def get_labels(self) -> list[int]:
        return self._labels

    def _sample_frames(self, paths: list[str]) -> list[str]:
        """Uniformly sample or pad paths to self.n_frames."""
        n = len(paths)
        if n == 0:
            return [""] * self.n_frames
        if n >= self.n_frames:
            indices = np.linspace(0, n - 1, self.n_frames, dtype=int)
            return [paths[i] for i in indices]
        else:
            # Pad by repeating the last frame
            return paths + [paths[-1]] * (self.n_frames - n)

    def _load_image(self, path: str) -> np.ndarray:
        if not path:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.imread(path)
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _apply_transform(self, rgb: np.ndarray) -> torch.Tensor:
        if self.use_albumentations:
            return self.transform(image=rgb)["image"]
        return self.transform(rgb)

    @staticmethod
    def _default_to_tensor(rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0


# ── Sampler utilities ─────────────────────────────────────────────────────────
def make_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Each class is sampled with equal probability per batch.

    Args:
        labels: List of integer labels (0 or 1).

    Returns:
        WeightedRandomSampler instance.
    """
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Data Module (high-level API) ───────────────────────────────────────────────
class DeepfakeDataModule:
    """
    High-level data module that creates DataLoaders for train/val/test splits.

    Usage:
        dm = DeepfakeDataModule(splits_dir='data/splits', dataset='faceforensics',
                                 batch_size=16, image_size=224)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
        test_loader  = dm.test_dataloader()
    """

    def __init__(
        self,
        splits_dir: str | Path,
        dataset: str = "faceforensics",
        mode: str = "image",           # 'image' | 'video'
        batch_size: int = 16,
        image_size: int = 224,
        n_frames: int = 32,
        augmentation_strength: str = "medium",
        num_workers: int = 4,
        pin_memory: bool = True,
        use_balanced_sampler: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ) -> None:
        self.splits_dir = Path(splits_dir)
        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_frames = n_frames
        self.augmentation_strength = augmentation_strength
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_balanced_sampler = use_balanced_sampler

        from preprocessing.augmentation import get_transforms
        self.train_tf, self.val_tf = get_transforms(image_size, augmentation_strength, mean, std)

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self) -> None:
        """Build datasets from manifest files."""
        DatasetClass = (
            DeepfakeImageDataset if self.mode == "image" else DeepfakeVideoDataset
        )
        kwargs = {}
        if self.mode == "video":
            kwargs["n_frames"] = self.n_frames

        def _manifest(split: str) -> Path:
            return self.splits_dir / f"{self.dataset}_{split}.json"

        self._train_ds = DatasetClass(
            _manifest("train"), transform=self.train_tf, **kwargs
        )
        self._val_ds = DatasetClass(
            _manifest("val"), transform=self.val_tf, **kwargs
        )
        self._test_ds = DatasetClass(
            _manifest("test"), transform=self.val_tf, **kwargs
        )
        logger.info(
            f"DataModule ready: "
            f"train={len(self._train_ds)}, "
            f"val={len(self._val_ds)}, "
            f"test={len(self._test_ds)}"
        )

    def train_dataloader(self) -> DataLoader:
        sampler = None
        shuffle = True
        if self.use_balanced_sampler:
            sampler = make_balanced_sampler(self._train_ds.get_labels())
            shuffle = False  # mutually exclusive with sampler
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
