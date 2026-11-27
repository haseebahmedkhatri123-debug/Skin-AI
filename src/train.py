"""Train a timm backbone on HAM10000.

Run from the project root:
    python src/train.py            # full run
    python src/train.py --quick    # 20% of the data, 3 epochs (pipeline smoke test)

Optional overrides: --backbone, --epochs, --img-size, --batch-size, --workers.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from dataset import SkinDataset, eval_transforms, train_transforms  # noqa: E402
from model import build_model, get_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train skin lesion classifier")
    p.add_argument("--quick", action="store_true",
                   help=f"Use {int(config.QUICK_FRACTION * 100)}%% of the data "
                        f"for {config.QUICK_EPOCHS} epochs")  # '%%' = literal % in argparse
    p.add_argument("--backbone", default=config.BACKBONE, choices=list(config.BACKBONES))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (auto if omitted)")
    return p.parse_args()


def subsample(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Stratified random subsample that keeps at least 2 images of every class."""
    parts = [g.sample(n=max(2, int(round(len(g) * frac))), random_state=config.SEED)
             for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=config.SEED)


def class_weights(targets: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, normalized so the mean weight is 1."""
    counts = np.bincount(targets, minlength=config.NUM_CLASSES).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class WeightedSmoothCE(nn.Module):
    """Class-weighted cross-entropy with label smoothing.

    Why not nn.CrossEntropyLoss(weight=..., label_smoothing=...)? PyTorch weights
    the smoothing term by the weights of *all* classes but normalizes by the
    target weights only. With strong imbalance weights this inflates the loss
    (e.g. 5.9 instead of ln 7 = 1.95 for uniform predictions) and pushes every
    prediction toward the rare classes. Here each sample's smoothed CE is
    weighted by its own class weight, and we take the weighted mean.
    """

    def __init__(self, weight: torch.Tensor, smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("weight", weight)
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_probs = logits.float().log_softmax(dim=1)
        n_classes = logits.size(1)
        smooth = torch.full_like(log_probs, self.smoothing / n_classes)
        smooth.scatter_(1, target.unsqueeze(1), 1 - self.smoothing + self.smoothing / n_classes)
        per_sample = -(smooth * log_probs).sum(dim=1)
        w = self.weight[target]
        return (per_sample * w).sum() / w.sum()


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Freeze/unfreeze everything except the classifier head."""
    head_params = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LR multiplier: linear warmup, then cosine decay to ~0."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * len(y))
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return (sum(losses) / len(targets),
            accuracy_score(targets, preds),
            balanced_accuracy_score(targets, preds))


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = get_device()
    epochs = args.epochs or (config.QUICK_EPOCHS if args.quick else config.EPOCHS)
    use_amp = config.USE_AMP and device.type == "cuda"
    # On CPU the workers compete with the model for cores, so keep them few.
    workers = args.workers if args.workers is not None else (
        min(8, os.cpu_count() or 2) if device.type == "cuda" else 2)
    print(f"Device: {device} | backbone: {args.backbone} | img {args.img_size} | "
          f"batch {args.batch_size} | epochs {epochs} | AMP {use_amp} | workers {workers}")

    # ---------------- data ----------------
    train_df, val_df = pd.read_csv(config.TRAIN_CSV), pd.read_csv(config.VAL_CSV)
    if args.quick:
        train_df = subsample(train_df, config.QUICK_FRACTION)
        val_df = subsample(val_df, config.QUICK_FRACTION)
        print(f"QUICK MODE: {len(train_df)} train / {len(val_df)} val images")

    train_ds = SkinDataset(None, train_transforms(args.img_size), df=train_df)
    val_ds = SkinDataset(None, eval_transforms(args.img_size), df=val_df)
    loader_kw = dict(num_workers=workers, pin_memory=device.type == "cuda",
                     persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    # ---------------- model / loss / optim ----------------
    model = build_model(args.backbone, pretrained=True).to(device)

    # Imbalance: class-weighted CE + label smoothing (no weighted sampler).
    weights = class_weights(train_ds.targets).to(device)
    print("Class weights:", {n: round(w, 2) for n, w in zip(config.CLASS_NAMES, weights.tolist())})
    criterion = WeightedSmoothCE(weights, config.LABEL_SMOOTHING).to(device)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(epochs * steps_per_epoch, config.WARMUP_EPOCHS * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.CLASSES_PATH.write_text(json.dumps(config.CLASS_NAMES, indent=2))

    # ---------------- loop ----------------
    best_bacc, bad_epochs, epoch_times = -1.0, 0, []
    log_every = max(1, steps_per_epoch // 10)
    for epoch in range(1, epochs + 1):
        frozen = epoch <= config.FREEZE_EPOCHS
        set_backbone_trainable(model, not frozen)
        if frozen:
            model.eval()                       # keep backbone norm stats fixed...
            model.get_classifier().train()     # ...but train the head
        else:
            model.train()

        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * len(y)
            seen += len(y)
            if step % log_every == 0 or step == steps_per_epoch:
                rate = seen / (time.time() - t0)
                print(f"  epoch {epoch} [{step}/{steps_per_epoch}] loss {running / seen:.4f} "
                      f"| {rate:.1f} img/s | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        val_loss, val_acc, val_bacc = evaluate(model, val_loader, val_criterion, device)
        epoch_times.append(time.time() - t0)
        eta = np.mean(epoch_times) * (epochs - epoch)
        print(f"Epoch {epoch}/{epochs}{' (backbone frozen)' if frozen else ''} | "
              f"train loss {running / seen:.4f} | val loss {val_loss:.4f} | "
              f"val acc {val_acc:.4f} | val bal-acc {val_bacc:.4f} | "
              f"time {fmt_time(epoch_times[-1])} | ETA {fmt_time(eta)}", flush=True)

        if val_bacc > best_bacc:
            best_bacc, bad_epochs = val_bacc, 0
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": args.backbone,
                "img_size": args.img_size,
                "classes": config.CLASS_NAMES,
                "epoch": epoch,
                "val_balanced_acc": val_bacc,
            }, config.BEST_MODEL_PATH)
            print(f"  -> new best (bal-acc {val_bacc:.4f}) saved to {config.BEST_MODEL_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping: no improvement for {bad_epochs} epochs.")
                break

    print(f"Done in {fmt_time(sum(epoch_times))}. Best val balanced accuracy: {best_bacc:.4f}")


if __name__ == "__main__":
    main()
