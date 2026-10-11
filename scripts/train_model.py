"""
scripts/train_model.py
-----------------------
Simplified training script that handles CPU training and produces
a trained model checkpoint for the Hybrid CNN-Transformer.

This script:
  1. Loads the dataset from data/splits/
  2. Creates train/val DataLoaders
  3. Trains the Hybrid model for the specified epochs
  4. Saves the best model checkpoint to checkpoints/hybrid/

Usage:
    python scripts/train_model.py --epochs 20 --batch_size 8
    python scripts/train_model.py --epochs 30 --batch_size 4  (low memory)
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import load_config
from models.hybrid.hybrid_model import HybridModel
from preprocessing.dataset_loader import DeepfakeImageDataset, make_balanced_sampler
from preprocessing.augmentation import get_transforms
from utils.device import get_device
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (use 4 for low memory)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_type", default="hybrid", choices=["hybrid", "cnn", "transformer"])
    parser.add_argument("--dataset", default="faceforensics")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    return parser.parse_args()


def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1."""
    preds = np.array(preds)
    labels = np.array(labels)
    
    acc = (preds == labels).mean()
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [TRAIN]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        acc = (preds == labels).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})
    
    avg_loss = total_loss / len(all_preds)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [VAL  ]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    
    avg_loss = total_loss / len(all_preds)
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = avg_loss
    
    # Compute AUC
    metrics["auc"] = compute_auc(all_labels, all_probs)
    return metrics


def compute_auc(labels, probs):
    """Compute ROC-AUC without sklearn."""
    pairs = sorted(zip(probs, labels), reverse=True)
    tp, fp = 0, 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp / max(total_pos, 1)
    return auc / max(total_neg, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    set_seed(42)
    device = get_device("auto")
    print(f"\n{'=' * 60}")
    print(f"  Deepfake Detection Model Training")
    print(f"{'=' * 60}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_type}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"{'=' * 60}\n")
    
    # ── Build transforms ──────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(
        image_size=cfg.data.image_size,
        strength="medium",
        mean=tuple(cfg.data.mean),
        std=tuple(cfg.data.std),
    )
    
    # ── Load datasets ─────────────────────────────────────────────────────────
    splits_dir = Path(cfg.data.splits_dir)
    
    from preprocessing.augmentation import ALBUMENTATIONS_AVAILABLE
    train_ds = DeepfakeImageDataset(
        manifest_path=splits_dir / f"{args.dataset}_train.json",
        transform=train_tf,
        use_albumentations=ALBUMENTATIONS_AVAILABLE,
    )
    val_ds = DeepfakeImageDataset(
        manifest_path=splits_dir / f"{args.dataset}_val.json",
        transform=val_tf,
        use_albumentations=ALBUMENTATIONS_AVAILABLE,
    )
    
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    
    # Balanced sampler
    sampler = make_balanced_sampler(train_ds.get_labels())
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.num_workers,
        pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=False,
    )
    
    # ── Build model ───────────────────────────────────────────────────────────
    if args.model_type == "hybrid":
        model = HybridModel(
            mode="image",
            backbone=cfg.model.cnn_backbone,
            pretrained=True,
            num_classes=2,
            embed_dim=cfg.model.transformer.embed_dim,
            num_heads=cfg.model.transformer.num_heads,
            num_layers=cfg.model.transformer.num_layers,
            mlp_ratio=cfg.model.transformer.mlp_ratio,
            dropout=cfg.model.transformer.dropout,
            freeze_cnn=False,
        )
    elif args.model_type == "cnn":
        from models.cnn.cnn_model import CNNModel
        model = CNNModel(pretrained=True)
    else:
        from models.transformer.vit_model import ViTModel
        model = ViTModel(pretrained=True)
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:  {total_params:,} total, {trainable_params:,} trainable\n")
    
    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    # Compute class weights
    labels = train_ds.get_labels()
    counts = np.bincount(labels)
    weights = 1.0 / (counts / counts.sum())
    weights = weights / weights.sum() * len(counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    
    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    history = []
    
    ckpt_dir = Path(cfg.training.checkpoint_dir) / args.model_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...\n")
    total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        epoch_time = time.time() - epoch_start
        
        # Log
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] | "
            f"lr={current_lr:.2e} | "
            f"T_loss={train_metrics['loss']:.4f} T_acc={train_metrics['accuracy']:.4f} | "
            f"V_loss={val_metrics['loss']:.4f} V_acc={val_metrics['accuracy']:.4f} "
            f"V_auc={val_metrics['auc']:.4f} V_f1={val_metrics['f1']:.4f} | "
            f"time={epoch_time:.0f}s"
        )
        
        # Save history
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "time_sec": epoch_time,
        }
        history.append(record)
        
        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            patience_counter = 0
            
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "config": dict(cfg),
            }
            best_path = ckpt_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"  >> New best model saved! AUC={best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    total_time = time.time() - total_start
    
    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    final_state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": val_metrics,
    }
    torch.save(final_state, ckpt_dir / f"{args.model_type}_final.pth")
    
    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Total time:    {total_time / 60:.1f} minutes")
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Best val AUC:  {best_val_auc:.4f}")
    print(f"  Best model:    {ckpt_dir / 'best_model.pth'}")
    print(f"  History:       {history_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
