"""
training/train_hybrid.py
--------------------------
Training script for the Hybrid CNN–Transformer model (main proposed model).

Supports two-phase training:
  Phase 1: Freeze CNN backbone, train only Transformer + head (fast convergence).
  Phase 2: Unfreeze CNN, fine-tune end-to-end with lower LR.

Usage:
    # Image mode
    python training/train_hybrid.py --mode image --dataset faceforensics

    # Video mode (temporal modeling)
    python training/train_hybrid.py --mode video --dataset faceforensics

    # Two-phase training
    python training/train_hybrid.py --mode image --two_phase

Options:
    --config     Path to config YAML
    --mode       image | video
    --dataset    faceforensics | dfdc | custom
    --lr         Learning rate
    --epochs     Number of epochs
    --batch_size Batch size
    --resume     Checkpoint path to resume
    --two_phase  Enable two-phase training (freeze then unfreeze CNN)
    --phase2_lr  Learning rate for phase 2 (default: lr * 0.1)
    --phase2_epochs  Epochs for phase 2 (default: epochs // 3)
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from configs import load_config
from models.hybrid.hybrid_model import HybridModel
from preprocessing.dataset_loader import DeepfakeDataModule
from training.trainer import Trainer
from utils.device import get_device
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__, log_dir="logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-Transformer")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--two_phase", action="store_true",
        help="Phase 1: frozen CNN. Phase 2: unfreeze and fine-tune end-to-end."
    )
    parser.add_argument("--phase2_lr", type=float, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def setup_data(cfg, args, mode: str) -> DeepfakeDataModule:
    dm = DeepfakeDataModule(
        splits_dir=cfg.data.splits_dir,
        dataset=args.dataset,
        mode=mode,
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        n_frames=cfg.data.frames_per_video,
        augmentation_strength=cfg.data.augmentation_strength,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        use_balanced_sampler=cfg.training.use_class_weights,
    )
    dm.setup()
    return dm


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels)
    weights = 1.0 / (counts / counts.sum())
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


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
    logger.info(f"Setting up data loaders (mode={args.mode}) ...")
    dm = setup_data(cfg, args, args.mode)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = None
    if cfg.training.use_class_weights:
        labels = dm._train_ds.get_labels()
        class_weights = compute_class_weights(labels)
        logger.info(f"Class weights: {class_weights.tolist()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — Frozen CNN backbone (Transformer + head only)
    # ══════════════════════════════════════════════════════════════════════════
    if args.two_phase:
        logger.info("=" * 50)
        logger.info("PHASE 1: Training with frozen CNN backbone")
        logger.info("=" * 50)

        p1_epochs = cfg.training.num_epochs - (args.phase2_epochs or cfg.training.num_epochs // 3)
        cfg.training["num_epochs"] = p1_epochs

        model = HybridModel(
            mode=args.mode,
            backbone=cfg.model.cnn_backbone,
            pretrained=not args.no_pretrained,
            num_classes=cfg.model.num_classes,
            embed_dim=cfg.model.transformer.embed_dim,
            num_heads=cfg.model.transformer.num_heads,
            num_layers=cfg.model.transformer.num_layers,
            mlp_ratio=cfg.model.transformer.mlp_ratio,
            dropout=cfg.model.transformer.dropout,
            n_frames=cfg.data.frames_per_video,
            freeze_cnn=True,   # Freeze CNN in phase 1
        )
        params = model.count_parameters()
        logger.info(f"Phase 1 trainable params: {params['trainable']:,} / {params['total']:,}")

        trainer_p1 = Trainer(
            model=model, config=cfg, device=device,
            model_name="hybrid_phase1", class_weights=class_weights,
        )
        trainer_p1.fit(dm.train_dataloader(), dm.val_dataloader(), resume_from=args.resume)

        # Load best phase 1 model to continue
        best_state = trainer_p1.checkpoint_manager.load_best()

        # ══════════════════════════════════════════════════════════════════════
        # Phase 2 — Unfreeze everything, fine-tune end-to-end with lower LR
        # ══════════════════════════════════════════════════════════════════════
        logger.info("=" * 50)
        logger.info("PHASE 2: Fine-tuning end-to-end (all layers)")
        logger.info("=" * 50)

        p2_epochs = args.phase2_epochs or cfg.training.num_epochs // 3
        p2_lr = args.phase2_lr or cfg.training.learning_rate * 0.1

        cfg.training["num_epochs"] = p2_epochs
        cfg.training["learning_rate"] = p2_lr
        cfg.training["warmup_epochs"] = 2
        cfg.training["early_stopping_patience"] = 8

        # Unfreeze CNN
        for p in model.cnn.backbone.parameters():
            p.requires_grad = True
        if best_state:
            model.load_state_dict(best_state["model_state_dict"])
        params = model.count_parameters()
        logger.info(f"Phase 2 trainable params: {params['trainable']:,} / {params['total']:,}")

        trainer_p2 = Trainer(
            model=model, config=cfg, device=device,
            model_name="hybrid", class_weights=class_weights,
        )
        trainer_p2.fit(dm.train_dataloader(), dm.val_dataloader())
        logger.info(f"Phase 2 best AUC: {trainer_p2.checkpoint_manager.best_metric:.4f}")

    else:
        # ── Single-phase training ──────────────────────────────────────────────
        logger.info("Building Hybrid CNN-Transformer model ...")
        model = HybridModel(
            mode=args.mode,
            backbone=cfg.model.cnn_backbone,
            pretrained=not args.no_pretrained,
            num_classes=cfg.model.num_classes,
            embed_dim=cfg.model.transformer.embed_dim,
            num_heads=cfg.model.transformer.num_heads,
            num_layers=cfg.model.transformer.num_layers,
            mlp_ratio=cfg.model.transformer.mlp_ratio,
            dropout=cfg.model.transformer.dropout,
            n_frames=cfg.data.frames_per_video,
            freeze_cnn=False,
        )
        params = model.count_parameters()
        logger.info(
            f"Parameters: {params['total']:,} total  "
            f"(CNN: {params['cnn']:,} | Transformer: {params['transformer']:,})"
        )

        trainer = Trainer(
            model=model, config=cfg, device=device,
            model_name="hybrid", class_weights=class_weights,
        )
        trainer.fit(
            dm.train_dataloader(), dm.val_dataloader(),
            resume_from=args.resume,
        )
        logger.info(
            f"Training complete! Best val AUC: "
            f"{trainer.checkpoint_manager.best_metric:.4f}"
        )
        logger.info(f"Best model: {trainer.checkpoint_manager.best_path}")


if __name__ == "__main__":
    main()
