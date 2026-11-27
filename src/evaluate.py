"""Evaluate the best checkpoint on the held-out test split.

Run from the project root:
    python src/evaluate.py

Prints accuracy, balanced accuracy and a per-class report, and saves
reports/confusion_matrix.png. Uses TTA (original + horizontal flip).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: we only save to file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,  # noqa: E402
                             classification_report, confusion_matrix)
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import SkinDataset, eval_transforms  # noqa: E402
from model import get_device, load_checkpoint, predict_tta  # noqa: E402


def plot_confusion_matrix(cm: np.ndarray, classes, path: Path):
    """Row-normalized confusion matrix (recall per class), annotated with counts."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("HAM10000 test set — confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color=color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    model, ckpt = load_checkpoint(config.BEST_MODEL_PATH, device)
    classes = ckpt["classes"]
    print(f"Loaded {ckpt['backbone']} (epoch {ckpt['epoch']}, "
          f"val bal-acc {ckpt['val_balanced_acc']:.4f}) on {device}")

    ds = SkinDataset(config.TEST_CSV, eval_transforms(ckpt["img_size"]))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    preds, targets = [], []
    for x, y in loader:
        preds.append(predict_tta(model, x.to(device)).argmax(1).cpu())
        targets.append(y)
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()

    labels = list(range(len(classes)))
    print(f"\nTest images: {len(targets)}")
    print(f"Accuracy:          {accuracy_score(targets, preds):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(targets, preds):.4f}\n")
    print(classification_report(targets, preds, labels=labels, target_names=classes,
                                digits=3, zero_division=0))

    cm = confusion_matrix(targets, preds, labels=labels)
    plot_confusion_matrix(cm, classes, config.CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
