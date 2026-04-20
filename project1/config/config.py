"""
Configuration file for Medical Image Classification Project
Central place for all hyperparameters and settings
"""

import os
from pathlib import Path

# Import utility function for auto-detecting class names
def _get_class_names_from_directory(data_dir: str):
    """Auto-detect class names from directory structure"""
    if not os.path.exists(data_dir):
        return []
    
    class_names = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    
    return sorted(class_names)

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = "/vast/projects/ebremer-group/fwang/image_classification/data"
TRAIN_DIR = DATA_DIR + "/train_split"
VAL_DIR = DATA_DIR + "/val_split"
TEST_DIR = DATA_DIR + "/test"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    directory1 = Path(directory)
    directory1.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

IMG_SIZE = 224                  # ResNet50 standard input size
BATCH_SIZE = 32                 # Batch size for training
RANDOM_SEED = 42                # For reproducibility

# Auto-detect class names from directory structure
_AUTO_CLASS_NAMES = _get_class_names_from_directory(str(TRAIN_DIR))
if _AUTO_CLASS_NAMES:
    CLASS_NAMES = _AUTO_CLASS_NAMES
    NUM_CLASSES = len(CLASS_NAMES)
else:
    # Fallback if directory not found (for config validation)
    CLASS_NAMES = [f"class_{i}" for i in range(10)]
    NUM_CLASSES = 10

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Learning rates
INITIAL_LEARNING_RATE = 1e-4    # Initial learning rate (frozen base)
FINE_TUNE_LEARNING_RATE = 1e-5  # Learning rate for fine-tuning

# Epochs
INITIAL_EPOCHS = 20             # Epochs with frozen base
FINE_TUNE_EPOCHS = 20           # Epochs for fine-tuning
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# Regularization
L2_REGULARIZATION = 1e-4        # L2 regularization parameter
DROPOUT_RATE = 0.5              # Dropout rate in custom layers

# Validation split
VALIDATION_SPLIT = 0.2          # 20% for validation, 80% for training

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

AUGMENTATION_CONFIG = {
    'rotation_range': 20,           # Random rotation
    'width_shift_range': 0.2,       # Random width shift
    'height_shift_range': 0.2,      # Random height shift
    'shear_range': 0.2,             # Shear transformation
    'zoom_range': 0.2,              # Zoom range
    'horizontal_flip': True,        # Horizontal flip
    'fill_mode': 'nearest',         # How to fill new pixels
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

FINE_TUNE_FROM_LAYER = 100     # Start fine-tuning from this layer
                                # (ResNet50 has ~175 layers)
DENSE_LAYER_1 = 512            # First dense layer units
DENSE_LAYER_2 = 256            # Second dense layer units

# ============================================================================
# CALLBACKS & REGULARIZATION
# ============================================================================

# Early Stopping
EARLY_STOPPING_PATIENCE = 5     # Stop if no improvement for N epochs
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MIN_DELTA = 0.001

# Learning Rate Reduction
LR_REDUCTION_FACTOR = 0.5       # Multiply LR by this
LR_REDUCTION_PATIENCE = 3       # Monitor patience
LR_REDUCTION_MIN_LR = 1e-7      # Minimum learning rate

# Model Checkpointing
CHECKPOINT_MONITOR = 'val_accuracy'

# ============================================================================
# CLASS WEIGHTS (for imbalanced data)
# ============================================================================
# Set to True to use class weights (auto-calculated from data)
# Set to False if data is balanced
USE_CLASS_WEIGHTS = True

# ============================================================================
# OUTPUT & VISUALIZATION
# ============================================================================

# Model saving
SAVE_MODEL = True
BEST_MODEL_NAME = "best_model_weights.h5"
FINAL_MODEL_NAME = "final_model.h5"

# Visualization
PLOT_DPI = 100
PLOT_FIGSIZE = (12, 8)
CONFUSION_MATRIX_FIGSIZE = (12, 10)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to track
TRACK_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc'
]

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

VERBOSE_LEVEL = 1              # Verbosity level (0, 1, or 2)
SAVE_TRAINING_HISTORY = True   # Save training history to JSON
SAVE_PREDICTIONS = True        # Save predictions to CSV

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# GPU settings
USE_GPU = True                  # Set to False to force CPU
GPU_MEMORY_FRACTION = 0.9       # GPU memory fraction to use

# Mixed precision training (for faster training on compatible GPUs)
USE_MIXED_PRECISION = True

# ============================================================================
# CLASS NAMES (Auto-detected from data/train/ subdirectories)
# ============================================================================
# Class names are automatically detected from the subfolder names in data/train/
# Example: If data/train/ contains [pneumonia/, normal/, covid/]
# Then CLASS_NAMES will be: ['covid', 'normal', 'pneumonia'] (sorted)
# 
# No manual configuration needed - just organize your images properly:
# data/train/
# ├── class_name_1/
# ├── class_name_2/
# └── class_name_3/
# 
# The class names will be detected automatically on startup

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def print_config():
    """Print all configuration settings"""
    print("\n" + "="*70)
    print("PROJECT CONFIGURATION")
    print("="*70)
    print(f"\nPaths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Training Data: {TRAIN_DIR}")
    print(f"  Validating Data: {VAL_DIR}")
    print(f"  Test Data: {TEST_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"\nImage Processing:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Number of Classes: {NUM_CLASSES}")
    print(f"  Classes: {', '.join(CLASS_NAMES)}")
    print(f"\nTraining:")
    print(f"  Initial Learning Rate: {INITIAL_LEARNING_RATE}")
    print(f"  Fine-tune Learning Rate: {FINE_TUNE_LEARNING_RATE}")
    print(f"  Initial Epochs: {INITIAL_EPOCHS}")
    print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
    print(f"  Total Epochs: {TOTAL_EPOCHS}")
    print(f"\nData Augmentation: Enabled")
    print(f"  Rotation: {AUGMENTATION_CONFIG['rotation_range']}°")
    print(f"  Shift: {AUGMENTATION_CONFIG['width_shift_range']*100}%")
    print(f"  Zoom: {AUGMENTATION_CONFIG['zoom_range']*100}%")
    print(f"\nClass Weights: {'Enabled' if USE_CLASS_WEIGHTS else 'Disabled'}")
    print(f"Validation Split: {VALIDATION_SPLIT*100}%")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config()
