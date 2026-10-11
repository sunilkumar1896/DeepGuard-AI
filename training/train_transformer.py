"""
training/train_transformer.py
------------------------------
Training script for the Vision Transformer (ViT) baseline model.

Usage:
    python training/train_transformer.py [options]

Options:
    --config      Path to config YAML
    --model_name  timm model name (e.g., vit_small_patch16_224)
    --lr          Learning rate
    --epochs      Number of epochs
    --batch_size  Batch size
    --resume      Checkpoint to resume from
    --dataset     faceforensics | dfdc | custom
    --no_pretrained  Train from scratch (not recommended)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from configs import load_config
from models.transformer.vit_model import ViTModel
from preprocessing.dataset_loader import DeepfakeDataModule
from training.trainer import Trainer
from utils.device import get_device
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__, log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT deepfake detector")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model_name", default="vit_small_patch16_224")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.lr:
        cfg.training["learning_rate"] = args.lr
    if args.epochs:
        cfg.training["num_epochs"] = args.epochs
    if args.batch_size:
        cfg.training["batch_size"] = args.batch_size

    set_seed(cfg.project.seed)
    device = get_device(cfg.inference.device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Setting up data loaders ...")
    dm = DeepfakeDataModule(
        splits_dir=cfg.data.splits_dir,
        dataset=args.dataset,
        mode="image",   # ViT uses single frames
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        augmentation_strength=cfg.data.augmentation_strength,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        use_balanced_sampler=cfg.training.use_class_weights,
    )
    dm.setup()

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Building ViT model: {args.model_name}")
    model = ViTModel(
        pretrained=not args.no_pretrained,
        model_name=args.model_name,
        image_size=cfg.data.image_size,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.transformer.embed_dim,
        num_heads=cfg.model.transformer.num_heads,
        num_layers=cfg.model.transformer.num_layers,
        mlp_ratio=cfg.model.transformer.mlp_ratio,
        dropout_rate=cfg.model.transformer.dropout,
    )
    params = model.count_parameters()
    logger.info(f"Parameters: {params['total']:,} total | Mode: {params['mode']}")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = None
    if cfg.training.use_class_weights:
        labels = dm._train_ds.get_labels()
        counts = np.bincount(labels)
        weights = 1.0 / (counts / counts.sum())
        weights = weights / weights.sum() * len(counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=cfg,
        device=device,
        model_name="transformer",
        class_weights=class_weights,
    )
    trainer.fit(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        resume_from=args.resume,
    )

    logger.info("Transformer training complete!")


if __name__ == "__main__":
    main()
