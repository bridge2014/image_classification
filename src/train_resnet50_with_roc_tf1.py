#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score

# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="TensorFlow ResNet50 with multi-class ROC/AUC.")
    p.add_argument("--data_dir", type=str, default="data", help="Root folder containing train/ and test/")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to save models/plots")
    p.add_argument("--img_size", type=int, default=224, help="Image size (height=width)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.2, help="Fraction of train/ used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_stage1", type=int, default=10, help="Epochs with base frozen")
    p.add_argument("--epochs_stage2", type=int, default=20, help="Epochs during fine-tuning")
    p.add_argument("--base_lr", type=float, default=1e-4, help="LR for stage 1")
    p.add_argument("--finetune_lr", type=float, default=1e-5, help="LR for stage 2 (fine-tuning)")
    p.add_argument("--fine_tune_at", type=int, default=140, help="Unfreeze from this layer index in ResNet50")
    return p.parse_args()

# =========================================================
# Utilities
# =========================================================
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()  # keep kernels fast

def configure(ds, training, seed):
    ds = ds.cache()
    if training:
        ds = ds.shuffle(1000, seed=seed)
    return ds.prefetch(tf.data.AUTOTUNE)

def plot_confusion_matrix(cm, classes, out_path, normalize=True, title="Confusion Matrix"):
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

def plot_multiclass_roc(y_true_oh, y_score, class_names, out_dir):
    """
    y_true_oh: (N, C) one-hot true labels (from test dataset)
    y_score:   (N, C) predicted probabilities (softmax)
    """
    os.makedirs(out_dir, exist_ok=True)
    num_classes = y_true_oh.shape[1]

    # Compute OvR ROC per class; skip classes absent in y_true
    fpr, tpr, roc_auc = {}, {}, {}
    present = []
    for i in range(num_classes):
        if y_true_oh[:, i].sum() == 0:
            continue
        present.append(i)
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    if present:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_oh[:, present].ravel(), y_score[:, present].ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    else:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = [np.array([0,1])]*2 + [np.nan]

    # macro-average
    if present:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in present]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in present:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(present)
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = [np.array([0,1])]*2 + [np.nan]

    # sklearn convenience (macro/weighted AUC)
    try:
        macro_auc = roc_auc_score(y_true_oh, y_score, multi_class="ovr", average="macro")
        weighted_auc = roc_auc_score(y_true_oh, y_score, multi_class="ovr", average="weighted")
    except Exception:
        macro_auc, weighted_auc = roc_auc.get("macro", np.nan), np.nan

    # Combined plot
    plt.figure(figsize=(10, 8))
    for i in present:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.55, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", linewidth=3,
             label=f"micro-average ROC (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color="navy", linestyle=":", linewidth=3,
             label=f"macro-average ROC (AUC={roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (OvR)")
    plt.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roc_multiclass.png")
    plt.savefig(out_path, dpi=160); plt.close()

    # Grid per-class
    cols = 5
    rows = int(np.ceil(max(1, len(present)) / cols))
    plt.figure(figsize=(3.2 * cols, 2.8 * rows))
    for idx, i in enumerate(present, start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(fpr[i], tpr[i], color="C0", lw=2, label=f"AUC={roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(class_names[i], fontsize=10)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.grid(alpha=0.2)
        if idx % cols == 1: ax.set_ylabel("TPR")
        if idx > (rows - 1) * cols: ax.set_xlabel("FPR")
        ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "roc_per_class_grid.png")
    plt.savefig(out_path2, dpi=160); plt.close()

    print("\nAUC summary (OvR):")
    for i in present:
        print(f"  {class_names[i]}: AUC = {roc_auc[i]:.4f}")
    print(f"  micro-average AUC = {roc_auc['micro']:.4f}")
    print(f"  macro-average AUC = {roc_auc['macro']:.4f}")
    print(f"  sklearn macro AUC = {macro_auc:.4f} | sklearn weighted AUC = {weighted_auc:.4f}")
    print(f"ROC plots saved to: {out_path} and {out_path2}")

# =========================================================
# Build datasets
# =========================================================
def build_datasets(data_dir, img_size, batch_size, val_split, seed):
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",               # one-hot for multi-class
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,        
        class_names=train_ds.class_names,       # ensure identical class order
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_names=train_ds.class_names,
    )

    return train_ds, val_ds, test_ds

# =========================================================
# Build model
# =========================================================
def build_model(img_size, num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    x = aug(inputs)
    x = layers.Lambda(preprocess_input, name="resnet50_preprocess")(x)

    base = ResNet50(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False

    x = base(x, training=False)  # keep BN frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="resnet50_tf_medical")
    return model, base

# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print("TensorFlow:", tf.__version__)

    # Datasets
    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir, args.img_size, args.batch_size, args.val_split, args.seed
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Performance
    train_ds = configure(train_ds, training=True, seed=args.seed)
    val_ds   = configure(val_ds, training=False, seed=args.seed)
    test_ds  = configure(test_ds, training=False, seed=args.seed)

    # Model
    model, base_model = build_model(args.img_size, num_classes)

    # Compile (stage 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
            tf.keras.metrics.AUC(name="auc_ovr", multi_label=True, num_labels=num_classes),
        ],
    )
    model.summary()

    # Callbacks
    ckpt_path = os.path.join(args.output_dir, "best_resnet50.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ]

    # Stage 1: train head (frozen base)
    print("\nStage 1: Training classifier head (frozen base)")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage1,
        callbacks=callbacks,
        verbose=1,
    )

    # Stage 2: fine-tune last blocks
    print("\nStage 2: Fine-tuning")
    for layer in base_model.layers[args.fine_tune_at:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.finetune_lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
            tf.keras.metrics.AUC(name="auc_ovr", multi_label=True, num_labels=num_classes),
        ],
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_stage2,
        callbacks=callbacks,
        verbose=1,
    )

    # -------- Test evaluation --------
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(test_ds, verbose=1)
    print(dict(zip(model.metrics_names, test_metrics)))

    # Collect predictions & labels
    y_true_oh = []
    y_pred = []
    y_score_chunks = []

    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0)
        y_score_chunks.append(probs)
        y_true_oh.append(batch_labels.numpy())
        y_pred.extend(np.argmax(probs, axis=1))

    y_score = np.vstack(y_score_chunks)         # (N, C)
    y_true_oh = np.vstack(y_true_oh)            # (N, C)
    y_true = np.argmax(y_true_oh, axis=1)       # (N,)

    # Classification report
    print("\nClassification Report (Test):")
    print(classification_report(y_true, np.array(y_pred), target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, np.array(y_pred), labels=np.arange(num_classes))
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, out_path=cm_path,
                          normalize=True, title="Normalized Confusion Matrix (Test)")
    print(f"Saved confusion matrix to {cm_path}")

    # ROC / AUC plots
    plot_multiclass_roc(y_true_oh, y_score, class_names, args.output_dir)

    # Save final model
    final_path = os.path.join(args.output_dir, "resnet50_final.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    

    print(" --- Save Accuracy Epoch chart ----")
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    plt.figure(figsize=(8, 8))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([args.epochs_stage1-1, args.epochs_stage1-1], plt.ylim(), label='Start Fine Tuning') # Mark the transition
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')    
    plt.grid(True, alpha=0.3)   
    out_path = os.path.join(args.output_dir, "accuracy_plot_roc_tf1.png") 
    plt.savefig(out_path, dpi=150, bbox_inches='tight')    

if __name__ == "__main__":
    main()