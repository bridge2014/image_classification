#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
###Dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib scikit-learn pillow

###How to Run
python train_resnet50_with_roc.py \
  --data_dir /vast/home/fwang/image_ai/data/ \
  --output_dir ../results/outputs \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 4
'''

import os
import time
import copy
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="ResNet50 training with ROC/AUC (multi-class).")
    p.add_argument("--data_dir", type=str, default="data", help="Root folder with train/ and test/")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to save models/plots")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2, help="Fraction of train as validation")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism toggles (slower but reproducible)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # keep True for speed


def plot_confusion_matrix(cm, classes, out_path, normalize=True, title="Confusion Matrix"):
    """Save a confusion matrix plot."""
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9,
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()


def plot_multiclass_roc(y_true, y_score, class_names, out_dir):
    """
    Compute and save per-class ROC curves (OvR), plus micro/macro averages.
    y_true: (N,) int labels
    y_score: (N, C) probabilities
    """
    os.makedirs(out_dir, exist_ok=True)
    classes = np.arange(len(class_names))
    y_true_bin = label_binarize(y_true, classes=classes)  # shape (N, C)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    present_classes = []
    for i in classes:
        # Skip classes that are not present in y_true (roc_curve requires both classes)
        if y_true_bin[:, i].sum() == 0:
            continue
        present_classes.append(i)
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average: compute fpr/tpr across all classes at once
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin[:, present_classes].ravel(),
                                              y_score[:, present_classes].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average: average the TPRs interpolated over all FPR points
    # Collect all FPR points
    all_fpr = np.unique(np.concatenate([fpr[i] for i in present_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in present_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= max(1, len(present_classes))
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Also compute macro/weighted AUC via sklearn convenience API
    try:
        macro_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        weighted_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
    except Exception:
        macro_auc = roc_auc["macro"]
        weighted_auc = np.nan

    # ----- Combined plot: micro/macro and per-class (fainter) -----
    plt.figure(figsize=(10, 8))
    # Per-class (faint)
    for i in present_classes:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    # Micro/macro highlighted
    plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", linewidth=3,
             label=f"micro-average ROC (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color="navy", linestyle=":", linewidth=3,
             label=f"macro-average ROC (AUC={roc_auc['macro']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (OvR)")
    plt.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    out_path = os.path.join(out_dir, "roc_multiclass.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    # ----- Grid of per-class ROC plots -----
    cols = 5
    rows = int(np.ceil(len(present_classes) / cols))
    plt.figure(figsize=(3.2 * cols, 2.8 * rows))
    for idx, i in enumerate(present_classes, start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(fpr[i], tpr[i], color="C0", lw=2, label=f"AUC={roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(class_names[i], fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.2)
        if idx % cols == 1:
            ax.set_ylabel("TPR")
        if idx > (rows - 1) * cols:
            ax.set_xlabel("FPR")
        ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path_grid = os.path.join(out_dir, "roc_per_class_grid.png")
    plt.savefig(out_path_grid, dpi=160)
    plt.close()

    # Print summary AUCs
    print("\nAUC summary (OvR):")
    for i in present_classes:
        print(f"  {class_names[i]}: AUC = {roc_auc[i]:.4f}")
    print(f"  micro-average AUC = {roc_auc['micro']:.4f}")
    print(f"  macro-average AUC = {roc_auc['macro']:.4f}")
    print(f"  sklearn macro AUC = {macro_auc:.4f} | sklearn weighted AUC = {weighted_auc:.4f}")


# =========================================================
# Training / Evaluation
# =========================================================
def train_model(model, criterion, optimizer, scheduler,
                train_loader, val_loader, device, epochs, out_dir):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            if phase == "val":
                scheduler.step(epoch_loss)

            print(f"{phase:>5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save best on val
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(out_dir, "best_resnet50.pth"))
                print("  ? Saved new best model")

    dt = time.time() - since
    print(f"\nTraining complete in {dt/60:.1f} min. Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    # Transforms
    IMG_SIZE = 224
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Build consistent splits (train/val from train/)
    base_ds = datasets.ImageFolder(train_dir)  # just to get indices/classes
    num_samples = len(base_ds)
    val_size = int(num_samples * args.val_split)
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(args.seed)).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ds = Subset(datasets.ImageFolder(train_dir, transform=transform_train), train_indices)
    val_ds   = Subset(datasets.ImageFolder(train_dir, transform=transform_eval), val_indices)
    test_ds  = datasets.ImageFolder(test_dir, transform=transform_eval)

    class_names = datasets.ImageFolder(train_dir).classes  # consistent mapping
    num_classes = len(class_names)
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model: ResNet50 transfer learning
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)

    # Train
    model = train_model(model, criterion, optimizer, scheduler,
                        train_loader, val_loader, device, args.epochs, out_dir)

    # -------- Test evaluation --------
    model.eval()
    y_true = []
    y_pred = []
    y_prob_chunks = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob_chunks.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.vstack(y_prob_chunks)  # shape (N, C)

    # Classification report
    print("\nClassification Report (Test):")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    plot_confusion_matrix(cm, class_names,
                          out_path=os.path.join(out_dir, "confusion_matrix.png"),
                          normalize=True,
                          title="Normalized Confusion Matrix (Test)")
    print(f"Saved confusion matrix to {os.path.join(out_dir, 'confusion_matrix.png')}")

    # -------- ROC/AUC (multi-class OvR) --------
    plot_multiclass_roc(y_true, y_score, class_names, out_dir)
    print(f"Saved ROC plots to {out_dir}")

    # Save final model (state dict)
    torch.save(model.state_dict(), os.path.join(out_dir, "resnet50_final.pth"))
    print(f"Saved final model to {os.path.join(out_dir, 'resnet50_final.pth')}")


if __name__ == "__main__":
    main()