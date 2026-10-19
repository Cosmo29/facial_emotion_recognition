"""Evaluate the trained model on the held-out test set: accuracy, per-class report, confusion matrix."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE, get_device, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--fear-boost", type=float, default=1.0, help="sensitivity multiplier for fear class prior")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    device = get_device()
    model, classes = load_model(args.ckpt, device)
    fear_idx = classes.index("fear") if "fear" in classes else -1

    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.ImageFolder(os.path.join(args.data, "test"), transform=tf)
    if ds.classes != classes:
        raise SystemExit(f"Class mismatch: checkpoint {classes} vs data {ds.classes}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            probs = torch.softmax(model(x.to(device)), dim=1)
            if args.fear_boost != 1.0 and fear_idx >= 0:
                probs[:, fear_idx] *= args.fear_boost
            y_pred.extend(probs.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    print(f"\nTest accuracy: {acc:.4f}  ({len(y_true)} images)\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test acc {acc:.3f})")
    fig.colorbar(im); fig.tight_layout()
    path = os.path.join(args.out, "confusion_matrix.png")
    fig.savefig(path, dpi=130)
    print(f"Saved {path}")

    # Export structured metrics tracking JSON
    import json
    from datetime import datetime
    report_dict = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    summary_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_images": len(y_true),
        "overall_accuracy": round(acc, 4),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": cm_norm.round(4).tolist(),
    }
    json_path = os.path.join(args.out, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved metrics summary to {json_path}")


if __name__ == "__main__":
    main()
