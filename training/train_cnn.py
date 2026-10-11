"""
training/train_cnn.py
----------------------
Training script for the CNN baseline model.

Usage:
    python training/train_cnn.py [options]

Options:
    --config     Path to config YAML (default: configs/config.yaml)
    --backbone   CNN backbone name (overrides config)
    --lr         Learning rate (overrides config)
    --epochs     Number of epochs (overrides config)
    --batch_size Batch size (overrides config)
    --resume     Path to checkpoint to resume from
    --dataset    Dataset name (faceforensics | dfdc | custom)
    --mode       image | video
"""
import argparse
import sys
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import load_config
from models.cnn.cnn_model import CNNModel
from preprocessing.dataset_loader import DeepfakeDataModule
from training.trainer import Trainer
from utils.device import get_device
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__, log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN deepfake detector")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load and optionally override config ───────────────────────────────────
    cfg = load_config(args.config)
    if args.backbone:
        cfg.model["cnn_backbone"] = args.backbone
    if args.lr:
        cfg.training["learning_rate"] = args.lr
    if args.epochs:
        cfg.training["num_epochs"] = args.epochs
    if args.batch_size:
        cfg.training["batch_size"] = args.batch_size
    if args.no_pretrained:
        cfg.model["pretrained"] = False

    set_seed(cfg.project.seed)
    device = get_device(cfg.inference.device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Setting up data loaders ...")
    dm = DeepfakeDataModule(
        splits_dir=cfg.data.splits_dir,
        dataset=args.dataset,
        mode=args.mode,
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        n_frames=cfg.data.frames_per_video,
        augmentation_strength=cfg.data.augmentation_strength,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        use_balanced_sampler=cfg.training.use_class_weights,
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Building CNN model: {cfg.model.cnn_backbone}")
    model = CNNModel(
        backbone=cfg.model.cnn_backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        dropout_rate=cfg.model.dropout_rate,
    )
    params = model.count_parameters()
    logger.info(f"Parameters: {params['total']:,} total / {params['trainable']:,} trainable")

    # ── Class weights for imbalanced data ─────────────────────────────────────
    class_weights = None
    if cfg.training.use_class_weights:
        labels = dm._train_ds.get_labels()
        counts = np.bincount(labels)
        weights = 1.0 / (counts / counts.sum())
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"Class weights: {class_weights.tolist()}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=cfg,
        device=device,
        model_name="cnn",
        class_weights=class_weights,
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=args.resume,
    )

    logger.info("CNN training complete!")
    logger.info(
        f"Best val AUC: {trainer.checkpoint_manager.best_metric:.4f}  "
        f"Best model: {trainer.checkpoint_manager.best_path}"
    )


if __name__ == "__main__":
    main()
