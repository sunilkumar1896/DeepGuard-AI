"""
training/trainer.py
---------------------
Reusable training engine for all three model architectures.

Features:
  - Mixed precision training (AMP)
  - Early stopping with configurable patience and metric
  - Learning rate scheduling (cosine, step, plateau, onecycle)
  - Gradient clipping
  - Best model checkpointing (top-K)
  - Resume from checkpoint
  - TensorBoard logging
  - Detailed per-epoch console output
  - Class-weighted loss for imbalanced datasets

Usage:
    from training.trainer import Trainer
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
"""
import json
import time
import math
from pathlib import Path
from typing import Optional
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.logger import get_logger
from utils.device import log_memory_usage

logger = get_logger(__name__)


# ── Metric tracker ─────────────────────────────────────────────────────────────
class MetricTracker:
    """Accumulate batch-level metrics and compute epoch averages."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0) + n

    def result(self) -> dict[str, float]:
        return {
            k: self._sums[k] / max(self._counts[k], 1)
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ── Checkpoint manager ─────────────────────────────────────────────────────────
class CheckpointManager:
    """
    Saves top-K checkpoints and always keeps the best model.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        model_name:     Used as filename prefix.
        top_k:          Max number of checkpoints to keep.
        mode:           'max' or 'min' for the monitored metric.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str,
        top_k: int = 3,
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.top_k = top_k
        self.mode = mode
        self._saved: deque[tuple[float, Path]] = deque()  # (metric, path)
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.best_path: Optional[Path] = None

    def save(
        self,
        epoch: int,
        metric: float,
        state: dict,
    ) -> bool:
        """
        Save checkpoint if it's in top-K.

        Returns:
            True if this is the new best checkpoint.
        """
        is_best = (
            (self.mode == "max" and metric > self.best_metric)
            or (self.mode == "min" and metric < self.best_metric)
        )

        path = self.checkpoint_dir / f"{self.model_name}_epoch{epoch:03d}_metric{metric:.4f}.pth"
        torch.save(state, path)
        self._saved.append((metric, path))

        # Remove oldest if over top_k limit
        while len(self._saved) > self.top_k:
            _, old_path = self._saved.popleft()
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()

        if is_best:
            self.best_metric = metric
            self.best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(state, self.best_path)
            logger.info(f"New best model saved: metric={metric:.4f}")

        return is_best

    def load_best(self) -> Optional[dict]:
        """Load the best checkpoint state dict."""
        if self.best_path and self.best_path.exists():
            return torch.load(self.best_path, map_location="cpu")
        return None


# ── Early Stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:   Epochs to wait before stopping.
        mode:       'max' (higher is better) or 'min'.
        min_delta:  Minimum change to count as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Call after each epoch.

        Returns:
            True if training should stop.
        """
        improved = (
            (self.mode == "max" and metric > self.best + self.min_delta)
            or (self.mode == "min" and metric < self.best - self.min_delta)
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} epochs without improvement."
            )

        return self.should_stop


# ── Scheduler factory ──────────────────────────────────────────────────────────
def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 5,
    **kwargs,
):
    """Build LR scheduler from name."""
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=kwargs.get("patience", 5),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 1e-3),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        raise ValueError(f"Unknown scheduler: '{scheduler_name}'")


# ── Warmup scheduler wrapper ───────────────────────────────────────────────────
class WarmupScheduler:
    """Linear warmup then hand off to base scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_scheduler,
        warmup_epochs: int,
        base_lr: float,
    ) -> None:
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self._epoch = 0

    def step(self, metric: float = None) -> None:
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self._epoch / max(1, self.warmup_epochs))
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            if isinstance(self.base_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Main Trainer class ─────────────────────────────────────────────────────────
class Trainer:
    """
    General-purpose training engine.

    Args:
        model:          PyTorch model.
        config:         DotDict config object.
        device:         torch.device.
        model_name:     Used for checkpoint naming (e.g., 'cnn', 'hybrid').
        class_weights:  Optional tensor of class weights for loss.
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
        model_name: str = "model",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_name = model_name

        cfg_tr = config.training

        # ── Loss ──────────────────────────────────────────────────────────────
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = self._build_optimizer()

        # ── Mixed Precision ───────────────────────────────────────────────────
        self.use_amp = cfg_tr.use_amp and device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # ── Checkpoint manager ─────────────────────────────────────────────────
        ckpt_dir = Path(cfg_tr.checkpoint_dir) / model_name
        self.checkpoint_manager = CheckpointManager(
            ckpt_dir,
            model_name=model_name,
            top_k=cfg_tr.keep_top_k,
            mode=cfg_tr.early_stopping_mode,
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        self.early_stopper = EarlyStopping(
            patience=cfg_tr.early_stopping_patience,
            mode=cfg_tr.early_stopping_mode,
        ) if cfg_tr.early_stopping else None

        # ── TensorBoard ────────────────────────────────────────────────────────
        self.writer = None
        if config.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = Path(config.logging.tensorboard_dir) / model_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(str(tb_dir))
            except Exception:
                self.writer = None

        # ── Training history ───────────────────────────────────────────────────
        self.history: list[dict] = []
        self._start_epoch = 1

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str | Path] = None,
    ) -> list[dict]:
        """
        Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            resume_from:  Path to checkpoint to resume from.

        Returns:
            Training history (list of per-epoch metric dicts).
        """
        cfg_tr = self.config.training

        # Build scheduler (needs steps_per_epoch)
        base_sched = build_scheduler(
            self.optimizer,
            scheduler_name=cfg_tr.scheduler,
            num_epochs=cfg_tr.num_epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=cfg_tr.warmup_epochs,
            min_lr=cfg_tr.min_lr,
            step_size=cfg_tr.step_size,
            gamma=cfg_tr.gamma,
            patience=cfg_tr.patience,
            max_lr=cfg_tr.learning_rate * 10,
        )
        self.scheduler = WarmupScheduler(
            self.optimizer, base_sched,
            warmup_epochs=cfg_tr.warmup_epochs,
            base_lr=cfg_tr.learning_rate,
        )

        # Resume if requested
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(f"Starting training: {self.model_name}")
        logger.info(f"  Epochs:       {cfg_tr.num_epochs}")
        logger.info(f"  Batch size:   {cfg_tr.batch_size}")
        logger.info(f"  Optimizer:    {cfg_tr.optimizer} (lr={cfg_tr.learning_rate})")
        logger.info(f"  Scheduler:    {cfg_tr.scheduler}")
        logger.info(f"  AMP:          {self.use_amp}")
        logger.info(f"  Device:       {self.device}")

        total_start = time.time()

        for epoch in range(self._start_epoch, cfg_tr.num_epochs + 1):
            epoch_start = time.time()

            # ── Train ──────────────────────────────────────────────────────────
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validate ───────────────────────────────────────────────────────
            val_metrics = self._val_epoch(val_loader, epoch)

            # ── LR step ────────────────────────────────────────────────────────
            monitor_val = val_metrics.get(cfg_tr.early_stopping_metric.replace("val_", ""), 0.0)
            self.scheduler.step(metric=monitor_val)
            current_lr = self.scheduler.get_last_lr()[0]

            # ── Log epoch ──────────────────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            epoch_record = {
                "epoch": epoch,
                "time_sec": epoch_time,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(epoch_record)

            self._log_epoch(epoch_record, cfg_tr.num_epochs)

            if self.writer:
                for k, v in epoch_record.items():
                    if isinstance(v, float):
                        self.writer.add_scalar(k, v, epoch)

            # ── Checkpoint ────────────────────────────────────────────────────
            monitor = epoch_record.get(cfg_tr.early_stopping_metric, 0.0)
            state = self._build_state(epoch, epoch_record)
            self.checkpoint_manager.save(epoch, monitor, state)

            # ── Early stopping ─────────────────────────────────────────────────
            if self.early_stopper and self.early_stopper.step(monitor):
                logger.info(f"Early stopping at epoch {epoch}.")
                break

            # ── Save every N epochs ────────────────────────────────────────────
            if epoch % cfg_tr.save_every_n_epochs == 0:
                periodic_path = (
                    Path(cfg_tr.checkpoint_dir) / self.model_name
                    / f"{self.model_name}_epoch{epoch:03d}_periodic.pth"
                )
                torch.save(state, periodic_path)

        total_time = time.time() - total_start
        logger.info(
            f"Training complete in {total_time / 60:.1f} min. "
            f"Best {cfg_tr.early_stopping_metric}: "
            f"{self.checkpoint_manager.best_metric:.4f}"
        )

        if self.writer:
            self.writer.close()

        self._save_history()
        return self.history

    # ── Training epoch ─────────────────────────────────────────────────────────
    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [TRAIN]", leave=False)
        for batch in pbar:
            images, labels = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (with optional AMP)
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
                self.optimizer.step()

            # OneCycleLR needs step per batch
            if isinstance(
                self.scheduler.base_scheduler,
                torch.optim.lr_scheduler.OneCycleLR,
            ):
                self.scheduler.base_scheduler.step()

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            tracker.update({"loss": loss.item(), "acc": acc}, n=images.size(0))

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.3f}"})

        metrics = tracker.result()
        metrics["f1"] = self._f1_score(all_preds, all_labels)
        return metrics

    # ── Validation epoch ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        tracker = MetricTracker()
        all_preds, all_labels, all_probs = [], [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [VAL  ]", leave=False)
        for batch in pbar:
            images, labels = self._unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

            tracker.update({"loss": loss.item(), "acc": acc}, n=images.size(0))
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        metrics = tracker.result()
        metrics["f1"] = self._f1_score(all_preds, all_labels)
        metrics["auc"] = self._roc_auc(all_labels, all_probs)
        return metrics

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _unpack_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack (images, labels) from a DataLoader batch."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        raise ValueError(f"Unexpected batch format: {type(batch)}")

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.config.training
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        name = cfg.optimizer.lower()
        if name == "adamw":
            return optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif name == "adam":
            return optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif name == "sgd":
            return optim.SGD(
                params, lr=cfg.learning_rate,
                momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: '{name}'")

    def _build_state(self, epoch: int, metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": dict(self.config),
        }

    def _load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._start_epoch = state["epoch"] + 1
        logger.info(f"Resumed from checkpoint: {path} (epoch {state['epoch']})")

    def _save_history(self) -> None:
        history_path = (
            Path(self.config.training.checkpoint_dir)
            / self.model_name
            / "training_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    @staticmethod
    def _log_epoch(record: dict, total_epochs: int) -> None:
        ep = record["epoch"]
        logger.info(
            f"Epoch [{ep:3d}/{total_epochs}] | "
            f"lr={record['lr']:.2e} | "
            f"T_loss={record['train_loss']:.4f} T_acc={record['train_acc']:.4f} | "
            f"V_loss={record['val_loss']:.4f} V_acc={record['val_acc']:.4f} "
            f"V_auc={record.get('val_auc', 0.0):.4f} | "
            f"time={record['time_sec']:.0f}s"
        )

    @staticmethod
    def _f1_score(preds: list[int], labels: list[int]) -> float:
        """Compute binary F1 score without sklearn."""
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-8)

    @staticmethod
    def _roc_auc(labels: list[int], probs: list[float]) -> float:
        """Compute ROC-AUC using trapezoidal rule without sklearn."""
        pairs = sorted(zip(probs, labels), reverse=True)
        tp, fp = 0, 0
        total_pos = sum(labels)
        total_neg = len(labels) - total_pos
        if total_pos == 0 or total_neg == 0:
            return 0.5
        auc = 0.0
        prev_fp = 0
        for _, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp / max(total_pos, 1)
        return auc / max(total_neg, 1)
