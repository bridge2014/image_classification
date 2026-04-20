"""
Configuration file for Medical Image Classification project
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset configurations
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)  # ResNet50 input size
RANDOM_SEED = 42

# Model configurations
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
INITIAL_EPOCHS = 10  # Train head before unfreezing
FINE_TUNE_EPOCHS = 40  # Fine-tune full model

# Data Augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
BRIGHTNESS_RANGE = [0.8, 1.2]
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
FILL_MODE = 'nearest'

# Training parameters
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
CLASS_WEIGHT_MODE = 'balanced'  # Option to compute class weights

# Model saving
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "final_model.h5")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "checkpoint.h5")

# Results paths
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.pkl")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curves.png")
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")
