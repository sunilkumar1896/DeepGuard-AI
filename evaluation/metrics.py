"""
evaluation/metrics.py
----------------------
Comprehensive evaluation metrics for deepfake detection.

Computes:
  - Accuracy, Precision, Recall, F1-score
  - ROC-AUC curve
  - Confusion matrix
  - Per-class breakdown
  - Video-level inference timing metrics

Usage:
    from evaluation.metrics import compute_all_metrics, ModelEvaluator
"""
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve, average_precision_score,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Core metric computation ────────────────────────────────────────────────────
def compute_all_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model",
) -> dict:
    """
    Compute the full set of classification metrics.

    Args:
        y_true:     True binary labels (0=real, 1=fake).
        y_pred:     Predicted class labels.
        y_prob:     Predicted fake probability scores.
        threshold:  Classification threshold.
        model_name: Name for display.

    Returns:
        Dict with all metrics and curve data.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {"model_name": model_name, "threshold": threshold}

    if SKLEARN_OK:
        metrics["accuracy"]  = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["auc"]       = float(roc_auc_score(y_true, y_prob))
        metrics["ap"]        = float(average_precision_score(y_true, y_prob))
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # Specificity (True Negative Rate)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = float(tn / max(tn + fp, 1))
            metrics["sensitivity"] = float(tp / max(tp + fn, 1))  # = recall
            metrics["fpr"] = float(fp / max(fp + tn, 1))

        # ROC curve data (for plotting)
        fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
        metrics["roc_fpr"] = fpr_arr.tolist()
        metrics["roc_tpr"] = tpr_arr.tolist()
        metrics["roc_thresholds"] = thresholds.tolist()

        # Precision-Recall curve
        prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_prob)
        metrics["pr_precision"] = prec_arr.tolist()
        metrics["pr_recall"] = rec_arr.tolist()

        # Per-class report
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        )
    else:
        # Fallback: manual computation
        metrics.update(_manual_metrics(y_true, y_pred, y_prob))

    # Summary counts
    metrics["n_samples"] = int(len(y_true))
    metrics["n_real"] = int((y_true == 0).sum())
    metrics["n_fake"] = int((y_true == 1).sum())

    return metrics


def _manual_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Manual metric computation without sklearn."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    # Simple AUC via trapezoidal approximation
    pairs = sorted(zip(y_prob.tolist(), y_true.tolist()), reverse=True)
    total_pos = int(y_true.sum())
    total_neg = int((y_true == 0).sum())
    auc = 0.0
    tpcount = 0
    for _, label in pairs:
        if label == 1:
            tpcount += 1
        else:
            auc += tpcount / max(total_pos, 1)
    auc /= max(total_neg, 1)

    return {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "specificity": tn / max(tn + fp, 1),
        "sensitivity": rec,
    }


def print_metrics_table(metrics_list: list[dict]) -> None:
    """Pretty-print a comparison table of multiple models."""
    header = f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}"
    print("\n" + "=" * len(header))
    print("  Model Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        row = (
            f"{m.get('model_name', '?'):<25} "
            f"{m.get('accuracy', 0):.4f}  "
            f"{m.get('precision', 0):.4f}  "
            f"{m.get('recall', 0):.4f}  "
            f"{m.get('f1', 0):.4f}  "
            f"{m.get('auc', 0):.4f}"
        )
        print(row)
    print("=" * len(header) + "\n")


# ── Model Evaluator ────────────────────────────────────────────────────────────
class ModelEvaluator:
    """
    Full evaluation pipeline for a single trained model.

    Handles:
      - Running inference on test DataLoader
      - Computing all metrics
      - Measuring inference speed (FPS, time/batch)
      - Saving results to JSON
    """

    def __init__(
        self,
        model,
        device,
        threshold: float = 0.5,
        results_dir: str | Path = "results",
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.device = device
        self.threshold = threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def evaluate(self, test_loader) -> dict:
        """
        Run full evaluation on a test DataLoader.

        Returns:
            Metrics dictionary.
        """
        import torch
        import torch.nn.functional as F

        self.model.eval()
        self.model.to(self.device)

        y_true, y_pred, y_prob = [], [], []
        inference_times = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device)

                t0 = time.perf_counter()
                logits = self.model(images)
                t1 = time.perf_counter()

                probs = F.softmax(logits, dim=1)[:, 1]
                preds = (probs >= self.threshold).long()

                inference_times.append((t1 - t0) / images.size(0))  # per sample
                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())

        metrics = compute_all_metrics(
            y_true, y_pred, y_prob,
            threshold=self.threshold,
            model_name=self.model_name,
        )

        # Timing metrics
        mean_time = float(np.mean(inference_times))
        metrics["inference_ms_per_sample"] = mean_time * 1000
        metrics["fps"] = 1.0 / max(mean_time, 1e-9)

        # Print
        self._print_results(metrics)

        # Save
        out_path = self.results_dir / f"{self.model_name}_metrics.json"
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in metrics.items()
                       if not isinstance(v, list) or len(v) < 100}, f, indent=2)
        logger.info(f"Metrics saved to {out_path}")

        return metrics

    def _print_results(self, m: dict) -> None:
        print(f"\n{'=' * 50}")
        print(f"  Evaluation Results: {m['model_name']}")
        print(f"{'=' * 50}")
        print(f"  Samples:    {m['n_samples']} (Real={m['n_real']}, Fake={m['n_fake']})")
        print(f"  Accuracy:   {m.get('accuracy', 0):.4f}")
        print(f"  Precision:  {m.get('precision', 0):.4f}")
        print(f"  Recall:     {m.get('recall', 0):.4f}")
        print(f"  F1-Score:   {m.get('f1', 0):.4f}")
        print(f"  ROC-AUC:    {m.get('auc', 0):.4f}")
        print(f"  Specificity:{m.get('specificity', 0):.4f}")
        print(f"  Inference:  {m.get('inference_ms_per_sample', 0):.2f} ms/sample  "
              f"({m.get('fps', 0):.1f} FPS)")
        print(f"{'=' * 50}\n")

        cm = m.get("confusion_matrix")
        if cm:
            print("  Confusion Matrix:")
            print(f"              Pred Real  Pred Fake")
            print(f"  True Real:    {cm[0][0]:8d}  {cm[0][1]:8d}")
            print(f"  True Fake:    {cm[1][0]:8d}  {cm[1][1]:8d}")
            print()
