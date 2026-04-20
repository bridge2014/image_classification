# Project Architecture and Workflow Guide

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEDICAL IMAGE CLASSIFICATION SYSTEM                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   RAW DATA      │  (Your medical images)
│   (JPG/PNG)     │
└────────┬────────┘
         │
         │ Organized into:
         │ data/train/class_0 ... class_9
         │ data/val/class_0 ... class_9
         │ data/test/class_0 ... class_9
         │
         ▼
┌─────────────────────────────────────┐
│      DATA LOADER (data_loader.py)   │
├─────────────────────────────────────┤
│ ✓ Load from directory structure    │
│ ✓ Resize to 224x224               │
│ ✓ Generate class weights          │
│ ─────────────────────────────────── │
│ TRAINING DATA:                      │
│  • Random rotation (±20°)          │
│  • Random shift (±20%)             │
│  • Random zoom (±20%)              │
│  • Brightness adjust (0.8-1.2x)    │
│  • Horizontal flip                 │
│ ─────────────────────────────────── │
│ VAL/TEST DATA:                      │
│  • Only rescaling (no augmentation)│
│ ─────────────────────────────────── │
│ CLASS WEIGHTS:                      │
│  • weight = total_samples /        │
│    (num_classes * samples_per_class)
└────────┬────────────────────────────┘
         │
         │ Batches: TensorFlow Dataset
         │ Size: 224x224x3 RGB
         │
         ▼
┌──────────────────────────────────────┐
│     MODEL (model.py)                 │
├──────────────────────────────────────┤
│                                      │
│  ResNet50 Architecture:              │
│  ┌────────────────────────────────┐ │
│  │ Input: 224x224x3 RGB Images    │ │
│  ├────────────────────────────────┤ │
│  │ Data Augmentation Layer        │ │
│  ├────────────────────────────────┤ │
│  │ Rescaling (1/127.5 - 1)        │ │
│  ├────────────────────────────────┤ │
│  │ ResNet50 Convolutional Stack   │ │
│  │ (2048 feature maps)            │ │
│  ├────────────────────────────────┤ │
│  │ Global Average Pooling         │ │
│  │ (Output: 2048)                 │ │
│  ├────────────────────────────────┤ │
│  │ Dense(256) + ReLU              │ │
│  │ Batch Normalization            │ │
│  │ Dropout(50%)                   │ │
│  ├────────────────────────────────┤ │
│  │ Dense(128) + ReLU              │ │
│  │ Batch Normalization            │ │
│  │ Dropout(30%)                   │ │
│  ├────────────────────────────────┤ │
│  │ Dense(10) + Softmax            │ │
│  │ (Output: Class probabilities)  │ │
│  └────────────────────────────────┘ │
│                                      │
│ Total Parameters: ~23.8 Million     │
│ - ResNet50: ~23.6 Million           │
│ - Custom Head: ~0.2 Million         │
└────────┬──────────────────────────┘
         │
         │ Loading pre-trained
         │ ImageNet weights
         │
         ▼
┌──────────────────────────────────────┐
│    TRAINING (train.py)               │
├──────────────────────────────────────┤
│                                      │
│  PHASE 1: Train Head (10 epochs)    │
│  ─────────────────────────────────  │
│  ResNet50 base: FROZEN ❄️            │
│  Custom head: TRAINABLE 🔥          │
│  Learning rate: 1e-3                │
│  Loss: Categorical Crossentropy    │
│  Optimizer: Adam                    │
│                                      │
│  Callback: ModelCheckpoint → best   │
│  Callback: EarlyStopping            │
│  Callback: ReduceLROnPlateau        │
│                                      │
│  Output: Adapted features on top   │
│  ─────────────────────────────────  │
│                                      │
│  PHASE 2: Fine-tune Full (40 epochs)│
│  ─────────────────────────────────  │
│  ResNet50 base: UNFROZEN (50 layers)│
│  All layers: TRAINABLE 🔥           │
│  Learning rate: 1e-5 (very small)  │
│  Loss: Categorical Crossentropy    │
│  Optimizer: Adam                    │
│                                      │
│  With class weights for imbalance   │
│                                      │
│  Callback: ModelCheckpoint → best   │
│  Callback: EarlyStopping            │
│  Callback: ReduceLROnPlateau        │
│  Callback: TensorBoard              │
│                                      │
│  Output: Fine-tuned medical model  │
│                                      │
└────────┬──────────────────────────┘
         │
         │ Saves:
         │ - models/best_model.h5
         │ - models/final_model.h5
         │ - results/training_history.pkl
         │
         ▼
┌──────────────────────────────────────┐
│    EVALUATION (evaluate.py)          │
├──────────────────────────────────────┤
│                                      │
│ Load trained model                  │
│ Generate predictions on test data   │
│                                      │
│ METRICS CALCULATED:                 │
│ ✓ Accuracy (overall)                │
│ ✓ Precision (macro & weighted)      │
│ ✓ Recall (macro & weighted)         │
│ ✓ F1-Score (macro & weighted)       │
│ ✓ ROC-AUC (per-class & averaged)   │
│                                      │
│ VISUALIZATIONS GENERATED:           │
│ ✓ Confusion Matrix Heatmap          │
│   Shows: True vs Predicted classes  │
│                                      │
│ ✓ ROC Curves (10 plots)             │
│   Shows: Per-class performance      │
│                                      │
│ REPORTS SAVED:                      │
│ ✓ Classification Report             │
│   (precision, recall, F1 per class) │
│                                      │
│ ✓ Metrics Summary                   │
│   (all metrics in one text file)   │
│                                      │
└────────┬──────────────────────────┘
         │
         │ Results saved in results/
         │ - confusion_matrix.png
         │ - roc_curves.png
         │ - classification_report.txt
         │ - metrics.txt
         │
         ▼
┌──────────────────────────────────────┐
│      RESULTS & INTERPRETATION        │
├──────────────────────────────────────┤
│                                      │
│ 1. CONFUSION MATRIX:                │
│    - Diagonal: Correct predictions  │
│    - Off-diagonal: Misclassifications│
│    - Bright spots: Problem areas    │
│                                      │
│ 2. PER-CLASS METRICS:               │
│    - Precision: Of predicted pos,   │
│      how many are actually pos?     │
│    - Recall: Of actual pos,         │
│      how many we found?             │
│    - F1: Balance between both       │
│    - Per-class ROC-AUC: 0.0-1.0    │
│                                      │
│ 3. OVERALL PERFORMANCE:             │
│    - Accuracy: All correct / all    │
│    - Macro Avg: Equal weight        │
│    - Weighted Avg: By class size    │
│                                      │
│ 4. CLASS BALANCE CHECK:             │
│    - ROC-AUC near 1.0: Excellent   │
│    - ROC-AUC near 0.5: Poor        │
│    - High variance: Imbalanced data │
│                                      │
└──────────────────────────────────────┘
```

## Data Flow: From Image to Prediction

```
Raw Medical Image
    |
    ├─→ Load (JPG/PNG)
    │
    ├─→ Resize (224 × 224)
    │
    ├─→ TRAINING DATA ONLY: Data Augmentation
    │   • Random rotation
    │   • Random shift
    │   • Random zoom
    │   • Brightness adjustment
    │   • Horizontal flip
    │
    ├─→ Normalize (rescale to [-1, 1])
    │
    ├─→ Batch (32 images together)
    │
    ├─→ ResNet50 Processing
    │   • Conv blocks extract features
    │   • Progressively lower resolution
    │   • Higher level features
    │   • Output: 2048 feature maps
    │
    ├─→ Global Average Pooling
    │   • Reduce 7×7×2048 → 2048
    │
    ├─→ Custom Head Classification
    │   • Dense(256) + ReLU
    │   • Dense(128) + ReLU
    │   • Dense(10) + Softmax
    │
    ├─→ Output: 10 Probabilities
    │   [0.01, 0.15, 0.70, 0.02, ...]
    │
    └─→ Prediction: Class with max probability
        Class 2 (70% confidence)
```

## Two-Phase Training Strategy

```
PHASE 1: HEAD TRAINING (Freezing Base)
═════════════════════════════════════════════════════════════════

Input medical images
        │
        ▼
    ResNet50 Base [FROZEN ❄️]
        │
        ├──→ Conv Block 1 (no updates)
        ├──→ Conv Block 2 (no updates)
        ├──→ Conv Block 3 (no updates)
        ├──→ Conv Block 4 (no updates)
        │
        ▼
    Custom Head [TRAINABLE 🔥]
        │
        ├─→ Dense(256) ← UPDATES
        ├─→ Dense(128) ← UPDATES
        └─→ Dense(10) Softmax ← UPDATES
        │
        ▼
    Predictions
        │
        ▼
    Compute Loss
        │
        ▼
    Backpropagation
        │
        └─→ Only updates custom head weights
            (ResNet50 stays frozen)

Why freeze?
- ResNet50 already knows general image features
- Only need to adapt top layers to medical data
- Prevents catastrophic forgetting
- Faster training with lower learning rate


PHASE 2: FINE-TUNING (Unfreezing Base)
═════════════════════════════════════════════════════════════════

Input medical images
        │
        ▼
    ResNet50 Base [UNFROZEN 🔥]
        │
        ├─→ Conv Block 1 (UPDATES) ← Fine-tuned
        ├─→ Conv Block 2 (UPDATES) ← Fine-tuned
        ├─→ Conv Block 3 (UPDATES) ← Fine-tuned
        ├─→ Conv Block 4 (UPDATES) ← Fine-tuned
        │
        ▼
    Custom Head [TRAINABLE 🔥]
        │
        ├─→ Dense(256) ← Updates
        ├─→ Dense(128) ← Updates
        └─→ Dense(10) Softmax ← Updates
        │
        ▼
    Predictions
        │
        ▼
    Compute Loss (with class weights)
        │
        ▼
    Backpropagation
        │
        └─→ Updates ALL weights
            (ResNet50 + Custom head)
            With VERY LOW learning rate (1e-5)

Why fine-tune at low learning rate?
- Carefully adapt pre-trained features
- Preserve learned general knowledge
- Small updates to medical features
- Prevent overfitting with medical data
```

## Class Weight Balancing

```
Example: Imbalanced Medical Dataset

Total training images: 1000
Number of classes: 10

WITHOUT class weights:
─────────────────────
Class 0 (disease_a):  400 images → 40% of training
Class 1 (disease_b):  300 images → 30% of training
Class 2 (disease_c):  150 images → 15% of training
Class 3 (disease_d):   80 images
Class 4 (disease_e):   40 images → Only 4% of training
...
Class 9 (disease_j):   10 images → Only 1% of training

Problem:
- Model biased toward major classes
- Minority classes ignored
- Poor performance on rare diseases


WITH class weights:
───────────────────
weight = total_samples / (num_classes * samples_per_class)
weight = 1000 / (10 * samples_per_class)

Class 0: weight = 1000 / (10 * 400) = 0.25 (downweight)
Class 4: weight = 1000 / (10 * 40) = 2.50 (upweight)
Class 9: weight = 1000 / (10 * 10) = 10.0 (heavily upweight)

Effect:
- Loss for Class 9 = loss_value * 10.0
- Loss for Class 0 = loss_value * 0.25
- Minority classes have bigger influence
- balanced training for all classes
```

## Model Evaluation Metrics Explained

```
CONFUSION MATRIX
════════════════════════════════════════════

        Predicted
       C0 C1 C2 ...
     ┌──┬──┬──┬──┐
A    │95│ 2│ 3│..│ ← True Class 0: 95 correct, 2→1, 3→2
c    ├──┼──┼──┼──┤
t    │ 1│98│ 1│..│ ← True Class 1: 98 correct, 1→0, 1→2
u    ├──┼──┼──┼──┤
a    │ 4│ 2│94│..│ ← True Class 2: 94 correct, 4→0, 2→1
l    └──┴──┴──┴──┘

Diagonal = Correct predictions
Off-diagonal = Misclassifications (confusions)


PER-CLASS METRICS
═════════════════════════════════════════════

True Positives (TP): Correctly predicted as positive
False Positives (FP): Incorrectly predicted as positive
False Negatives (FN): Incorrectly predicted as negative

For Class 2:
TP = 94 (correctly predicted as class 2)
FP = 4 + 2 = 6 (others predicted as class 2)
FN = 3 + 1 = 4 (class 2 predicted as other)

Precision = TP / (TP + FP) = 94 / 100 = 0.94
  "Of what we predicted as class 2, 94% are correct"

Recall = TP / (TP + FN) = 94 / 98 = 0.96
  "Of actual class 2 instances, we found 96%"

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
F1 = 2 * (0.94 * 0.96) / (0.94 + 0.96) = 0.95
  "Balanced measure of precision & recall"


ROC-AUC CURVE
═════════════════════════════════════════════

True Positive Rate (TPR/Sensitivity)
= TP / (TP + FN) → What % of real positives we catch

False Positive Rate (FPR)
= FP / (FP + TN) → What % of negatives we misclassify

AUC (Area Under Curve):
- 1.0 = Perfect classifier
- 0.5 = Random guessing
- 0.0 = Opposite predictions

Interpretation:
- AUC > 0.9: Excellent
- AUC > 0.8: Good
- AUC > 0.7: Fair
- AUC = 0.5: Random
- AUC < 0.5: Worse than random (check labels!)
```

## Performance Optimization Pyramid

```
┌─────────────────────────────────┐
│  FAST INFERENCE DEPLOYMENT      │  Milliseconds per image
├─────────────────────────────────┤
│ - Model quantization            │
│ - Edge deployment (TensorFlow   │
│   Lite, ONNX)                   │
│ - GPU inference                 │
└─────────────────────────────────┘
                 △
                / \
               /   \
              / TOP \
             /PRIORITY\
┌───────────────────────────────┐
│   BETTER ACCURACY             │  Validation metrics
├───────────────────────────────┤
│ - More training data          │
│ - Better data augmentation    │
│ - Tune hyperparameters        │
│ - Ensemble models             │
└───────────────────────────────┘
                 △
                / \
               /   \
              / HIGH \
             /PRIORITY\
┌──────────────────────────────────┐
│  PRODUCTION RELIABILITY          │  Error handling
├──────────────────────────────────┤
│ - Logging & monitoring          │
│ - Error handling                │
│ - Input validation              │
│ - Model versioning              │
│ - A/B testing framework         │
└──────────────────────────────────┘
```

## Troubleshooting Decision Tree

```
                    Problem?
                       │
                       ▼
        ┌──────────────────────────┐
        │ During Training?         │
        └──────────────────────────┘
              Yes│         │No
                 ▼         ▼
          Training issue  Evaluation/Inference
                 │                │
        ┌────────┴────────┐       ▼
        ▼                 ▼    Check test data
   Loss not          Slow
   decreasing        training
        │                │
        ├→ Check LR   ├→ Reduce batch
        ├→ Check data ├→ Use GPU
        ├→ Check model├→ Reduce epochs
        └→ More data  └→ Check RAM

        ▼                 ▼
    Poor accuracy      No predictions
        │                │
        ├→ More data  ├→ Check model path
        ├→ Tune LR    ├→ Check image format
        ├→ Check aug  ├→ Check image size
        └→ New arch   └→ Load test works?
```

## Files and Their Responsibilities

```
config.py
  ├─ All file paths (DATA_DIR, MODELS_DIR, etc)
  ├─ Hyperparameters (BATCH_SIZE, EPOCHS, etc)
  ├─ Augmentation parameters
  └─ All settings (change here for customization)

data_loader.py
  ├─ Load images from directories
  ├─ Apply data augmentation (training only)
  ├─ Calculate class weights
  └─ Create TensorFlow datasets

model.py
  ├─ ResNet50 architecture setup
  ├─ Custom head layers
  ├─ Model compilation
  ├─ Callbacks definition
  └─ Fine-tuning logic

train.py
  ├─ Orchestrates entire training
  ├─ Phases 1 and 2 of training
  ├─ Saves models
  └─ Saves training history

evaluate.py
  ├─ Load trained model
  ├─ Generate predictions
  ├─ Calculate all metrics
  ├─ Create visualizations
  └─ Save reports

train_model.py (main entry point)
  ├─ User interface for training
  ├─ Verifies data structure
  └─ Calls ModelTrainer class

evaluate_model.py (main entry point)
  ├─ User interface for evaluation
  ├─ Checks model exists
  └─ Calls ModelEvaluator class
```

---

**Understand this architecture and you understand the entire medical image classification system!**
