"""Train the emotion CNN on FER-2013 (folder layout: DATA/train/<emotion>/*.jpg, DATA/test/<emotion>/*.jpg)."""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from model import EmotionCNN, IMG_SIZE, get_device


def build_loaders(data_dir, batch_size, val_frac, workers, seed):
    norm = transforms.Normalize((0.5,), (0.5,))
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    train_full = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_full = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=eval_tf)
    classes = train_full.classes

    n_val = int(len(train_full) * val_frac)
    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val = random_split(range(len(train_full)), [len(train_full) - n_val, n_val], generator=g)
    train_ds = Subset(train_full, idx_train.indices)
    val_ds = Subset(val_full, idx_val.indices)

    targets = np.array(train_full.targets)[idx_train.indices]
    counts = np.bincount(targets, minlength=len(classes))
    # Softened inverse-frequency weights (square root): full inverse weights made 'disgust' ~16x heavier
    # than 'happy' and the model collapsed onto that one class early in training.
    weights = (counts.sum() / (len(classes) * np.maximum(counts, 1))) ** 0.5

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    return train_dl, val_dl, classes, torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
            if train and i % 25 == 0:
                print(f"  batch {i}/{len(loader)} | running loss {total_loss / n:.3f} acc {correct / n:.3f}", flush=True)
    return total_loss / n, correct / n


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


class FocalLoss(nn.Module):
    """Focal Loss: dynamically scales cross-entropy based on prediction confidence to focus on hard classes."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="folder containing train/ and test/")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=64, help="base channels; 32 is ~4x faster on CPU")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["ce", "focal"], default="ce", help="loss criterion (ce or focal)")
    p.add_argument("--gamma", type=float, default=2.0, help="focal loss focusing parameter")
    p.add_argument("--fear-boost", type=float, default=1.5, help="loss penalty multiplier for fear class")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    train_dl, val_dl, classes, class_w = build_loaders(
        args.data, args.batch_size, args.val_frac, args.workers, args.seed)
    if args.fear_boost != 1.0 and "fear" in classes:
        fear_idx = classes.index("fear")
        class_w[fear_idx] *= args.fear_boost
        print(f"Applied Fear loss boost {args.fear_boost:.1f}x -> class weight: {class_w[fear_idx]:.3f}")
    print(f"Classes: {classes} | train batches: {len(train_dl)} | val batches: {len(val_dl)}")

    model = EmotionCNN(num_classes=len(classes), width=args.width).to(device)
    if args.loss == "focal":
        print(f"Using Focal Loss (gamma={args.gamma}, label_smoothing=0.05)")
        criterion = FocalLoss(weight=class_w.to(device), gamma=args.gamma, label_smoothing=0.05)
    else:
        print("Using Class-Weighted Cross-Entropy Loss (label_smoothing=0.05)")
        criterion = nn.CrossEntropyLoss(weight=class_w.to(device), label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    metrics_records = []
    best_acc, best_path = 0.0, os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer, scheduler)
        vl, va = run_epoch(model, val_dl, criterion, device)
        dur = round(time.time() - t0, 1)
        curr_lr = optimizer.param_groups[0]["lr"]

        for k, v in zip(hist, [tl, ta, vl, va]):
            hist[k].append(v)
            
        metrics_records.append({
            "epoch": epoch,
            "train_loss": round(tl, 4),
            "train_acc": round(ta, 4),
            "val_loss": round(vl, 4),
            "val_acc": round(va, 4),
            "learning_rate": f"{curr_lr:.6f}",
            "time_sec": dur
        })

        flag = ""
        if va > best_acc:
            best_acc = va
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "val_acc": va, "epoch": epoch, "width": args.width}, best_path)
            flag = "  <- saved best"
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {dur:.0f}s{flag}")

    # Save training curves and CSV metrics history
    plot_history(hist, os.path.join(args.out, "training_curves.png"))
    
    import csv
    csv_path = os.path.join(args.out, "training_history.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "time_sec"])
        writer.writeheader()
        writer.writerows(metrics_records)
    print(f"Saved training metrics history to {csv_path}")

    print(f"\nDone. Best val accuracy: {best_acc:.3f}. Model saved to {best_path}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
