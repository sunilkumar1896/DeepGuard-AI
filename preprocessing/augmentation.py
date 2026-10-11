"""
preprocessing/augmentation.py
-------------------------------
Data augmentation pipeline for deepfake detection training.

Three augmentation strengths:
  - light:  Safe, label-preserving transforms only.
  - medium: Adds color jitter, mild geometric transforms.
  - heavy:  Aggressive augmentation for regularization.

All transforms use albumentations for speed and consistency.

Usage:
    from preprocessing.augmentation import get_transforms
    train_tf, val_tf = get_transforms(image_size=224, strength='medium')
"""
import cv2
import numpy as np
from typing import Callable

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import torchvision.transforms as T
    import torch
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def get_transforms(
    image_size: int = 224,
    strength: str = "medium",
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> tuple:
    """
    Build train and validation transform pipelines.

    Args:
        image_size: Target image size (square).
        strength:   'light' | 'medium' | 'heavy'.
        mean:       Normalization mean (ImageNet defaults).
        std:        Normalization std (ImageNet defaults).

    Returns:
        (train_transform, val_transform) — albumentations Compose objects
        that return {'image': tensor, 'label': label}.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return _torchvision_fallback(image_size, mean, std)

    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    if strength == "light":
        train_transform = A.Compose([
            A.Resize(image_size + 20, image_size + 20),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif strength == "medium":
        train_transform = A.Compose([
            A.Resize(image_size + 32, image_size + 32),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            ], p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),
            A.OneOf([
                A.GridDistortion(p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
            ], p=0.2),
            # JPEG compression artifacts (important for deepfake detection!)
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(4, 16),
                hole_width_range=(4, 16),
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif strength == "heavy":
        train_transform = A.Compose([
            A.Resize(image_size + 48, image_size + 48),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.6
            ),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=20),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(20.0, 100.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
            ], p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.ElasticTransform(alpha=2, sigma=50),
                A.Perspective(scale=(0.05, 0.1)),
            ], p=0.35),
            A.ImageCompression(quality_lower=40, quality_upper=100, p=0.6),
            A.CoarseDropout(
                num_holes_range=(2, 12),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                p=0.4,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unknown strength: '{strength}'. Use light|medium|heavy.")

    return train_transform, val_transform


def _torchvision_fallback(
    image_size: int, mean: tuple, std: tuple
) -> tuple:
    """Fallback transforms using torchvision if albumentations is unavailable."""
    if not TORCHVISION_AVAILABLE:
        raise ImportError(
            "Neither albumentations nor torchvision is installed. "
            "Install at least one: pip install albumentations"
        )

    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def apply_transform(
    image: np.ndarray,
    transform,
    use_albumentations: bool = True,
):
    """
    Apply a transform to a BGR numpy image.

    Args:
        image:              BGR uint8 numpy array.
        transform:          Albumentations Compose OR torchvision transform.
        use_albumentations: Whether transform is an albumentations pipeline.

    Returns:
        Transformed tensor (C, H, W) float32.
    """
    if use_albumentations and ALBUMENTATIONS_AVAILABLE:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = transform(image=rgb)
        return result["image"]
    else:
        # torchvision expects (H, W, C) RGB uint8
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(rgb)
