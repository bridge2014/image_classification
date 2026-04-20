"""
Diagnostics Script
Analyzes data distribution, model configuration, and training readiness
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config.config import (
    TRAIN_DIR, TEST_DIR, NUM_CLASSES, CLASS_NAMES, BATCH_SIZE,
    INITIAL_EPOCHS, FINE_TUNE_EPOCHS, USE_FOCAL_LOSS, 
    FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA, AUGMENTATION_CONFIG
)
from src.utils import get_class_distribution, get_class_weights


def analyze_class_distribution():
    """Analyze class imbalance severity"""
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION & IMBALANCE ANALYSIS")
    print("="*70)
    
    class_dist = get_class_distribution(TRAIN_DIR)
    
    if not class_dist:
        print("[ERROR] Could not load class distribution")
        return
    
    # Get counts
    counts = np.array(list(class_dist.values()))
    class_names_list = sorted(class_dist.keys())
    
    total_samples = counts.sum()
    mean_count = counts.mean()
    min_count = counts.min()
    max_count = counts.max()
    
    print(f"\nTotal Training Samples: {total_samples:,}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Average Samples per Class: {mean_count:.0f}")
    print(f"Min Samples: {min_count:,} ({class_names_list[np.argmin(counts)]})")
    print(f"Max Samples: {max_count:,} ({class_names_list[np.argmax(counts)]})")
    
    # Imbalance ratio
    imbalance_ratio = max_count / min_count
    print(f"\n??  Imbalance Ratio (Max/Min): {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 10:
        print("   ??  SEVERE imbalance detected! Consider using Focal Loss.")
    elif imbalance_ratio > 3:
        print("   ??  MODERATE imbalance detected. Focal Loss is recommended.")
    else:
        print("   ? Class distribution is relatively balanced.")
    
    # Class weights analysis
    print("\n[CLASS WEIGHTS] Inverse Frequency Weighting:")
    class_weights = get_class_weights(TRAIN_DIR)
    if class_weights:
        for idx, cls_name in enumerate(sorted(class_dist.keys())):
            weight = class_weights.get(idx, 1.0)
            count = class_dist[cls_name]
            pct = (count / total_samples) * 100
            print(f"  {cls_name:35s}: count={count:6,} ({pct:5.2f}%)  weight={weight:.4f}")


def analyze_model_config():
    """Analyze model and training configuration"""
    print("\n" + "="*70)
    print("MODEL & TRAINING CONFIGURATION")
    print("="*70)
    
    print(f"\n[ARCHITECTURE]")
    print(f"  Base Model: ResNet50 (ImageNet pretrained)")
    print(f"  Input Size: 224 x 224")
    print(f"  Output Classes: {NUM_CLASSES}")
    print(f"  Custom Layers: 1024 ? 512 ? 256 ? {NUM_CLASSES}")
    
    print(f"\n[LOSS FUNCTION]")
    if USE_FOCAL_LOSS:
        print(f"  Type: Focal Loss (for class imbalance)")
        print(f"  Gamma: {FOCAL_LOSS_GAMMA} (focusing parameter)")
        print(f"  Alpha: {FOCAL_LOSS_ALPHA} (balance parameter)")
        print(f"  ? Optimal for severe class imbalance (current dataset)")
    else:
        print(f"  Type: Categorical Crossentropy")
        print(f"  ??  Consider enabling Focal Loss for imbalanced data")
    
    print(f"\n[TRAINING SCHEDULE]")
    print(f"  Phase 1 (Frozen Base): {INITIAL_EPOCHS} epochs")
    print(f"  Phase 2 (Fine-tune): {FINE_TUNE_EPOCHS} epochs")
    print(f"  Total Epochs: {INITIAL_EPOCHS + FINE_TUNE_EPOCHS}")
    
    print(f"\n[DATA AUGMENTATION]")
    print(f"  Rotation: {AUGMENTATION_CONFIG.get('rotation_range', 20)}")
    print(f"  Shift (H/V): {AUGMENTATION_CONFIG.get('width_shift_range', 0.2)}")
    print(f"  Zoom: {AUGMENTATION_CONFIG.get('zoom_range', 0.2)}")
    print(f"  Horizontal Flip: {AUGMENTATION_CONFIG.get('horizontal_flip', True)}")
    print(f"  Brightness Range: {AUGMENTATION_CONFIG.get('brightness_range', 'N/A')}")
    print(f"  ? Enhanced for medical imaging")


def estimate_training_time():
    """Estimate training time"""
    print("\n" + "="*70)
    print("TRAINING TIME ESTIMATE")
    print("="*70)
    
    from src.utils import get_class_distribution
    
    class_dist = get_class_distribution(TRAIN_DIR)
    total_samples = sum(class_dist.values())
    
    # Estimate based on dataset size
    steps_per_epoch = int(np.ceil(total_samples * 0.8 / BATCH_SIZE))
    total_steps = steps_per_epoch * (INITIAL_EPOCHS + FINE_TUNE_EPOCHS)
    
    # Very rough estimate: ~0.5-1.0 seconds per step on GPU
    estimated_time_min = total_steps * 0.01  # minutes (conservative estimate)
    estimated_time_hour = estimated_time_min / 60
    
    print(f"\nTraining Data: {total_samples:,} samples")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Steps per Epoch: ~{steps_per_epoch:,}")
    print(f"Total Steps: ~{total_steps:,}")
    print(f"\nEstimated Training Time: ~{estimated_time_hour:.1f} hours (GPU)")
    print(f"                         ~{estimated_time_hour*3:.1f} hours (CPU - not recommended)")

# -*- coding: utf-8 -*-
def print_recommendations():
    """Print recommendations for improving accuracy"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR ACCURACY IMPROVEMENT")
    print("="*70)
    
    print("""
1. CLASS IMBALANCE HANDLING (CRITICAL):
   ? Focal Loss is enabled - focuses learning on hard-to-classify samples
   ? Class weights are auto-calculated - handles frequency imbalance
   ? Consider collecting more data for minority classes (Nerves, Normal ductal epithelium)

2. MODEL & ARCHITECTURE:
   ? Using ResNet50 with enhanced custom layers (1024 ? 512 ? 256)
   ? Strong regularization with Dropout and L2 penalties
   Tip: Consider using a medical-specific backbone if available (e.g., models trained on PathImageNet)

3. DATA AUGMENTATION:
   ? Enhanced augmentation strategy is enabled
   Tip: Monitor that augmentation is helping (check training plots)
   Tip: For medical images, domain-specific augmentation may help more

4. TRAINING STRATEGY:
   ? Two-phase training approach (frozen base ? fine-tuning)
   ? Progressive unfreezing from layer 100
   Tip: Monitor convergence - if stuck, try higher initial learning rate

5. MONITORING & DEBUGGING:
    Enable TensorBoard: tensorboard --logdir=logs/
    Check training_history plots for signs of:
      - Overfitting (val_loss rising while train_loss falling)
      - Poor class-specific performance (check per-class metrics)
      - Learning rate issues (plateaued loss)

6. NEXT STEPS IF ACCURACY STILL LOW:
   • Verify image preprocessing matches expected format
   • Analyze hard-to-classify tissue types
   • Try ensemble methods with multiple model architectures
   • Consider Semi-Supervised Learning if you have unlabeled data
    """)


def main():
    """Run full diagnostics"""
    print("\n" + "¦"*70)
    print("¦" + " "*68 + "¦")
    print("¦" + "  MEDICAL IMAGE CLASSIFICATION - TRAINING DIAGNOSTICS".center(68) + "¦")
    print("¦" + " "*68 + "¦")
    print("¦"*70)
    
    try:
        analyze_class_distribution()
        analyze_model_config()
        estimate_training_time()
        print_recommendations()
        
        print("\n" + "="*70)
        print("? DIAGNOSTICS COMPLETE - Ready to train!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()