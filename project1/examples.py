"""
Complete Example Usage
Demonstrates all components of the medical imaging project
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, INITIAL_EPOCHS, FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE, VALIDATION_SPLIT,
    TRAIN_DIR, TEST_DIR, MODELS_DIR, RESULTS_DIR, CLASS_NAMES
)
from src import (
    ResNet50Classifier, DataLoader, Evaluator,
    get_class_weights, get_class_distribution, verify_data_structure,
    plot_class_distribution
)


def example_1_data_analysis():
    """
    Example 1: Analyze training data distribution
    
    NOTE: Class names and NUM_CLASSES are automatically detected from
    the subdirectory names in TRAIN_DIR during config initialization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: DATA ANALYSIS (Classes Auto-Detected)")
    print("="*70)
    
    print(f"\nClasses auto-detected from: {TRAIN_DIR}")
    print(f"Detected classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
    
    # Verify data structure
    print("\nVerifying data structure...")
    verify_data_structure(TRAIN_DIR, TEST_DIR)
    
    # Get class distribution from training data only
    print("\nAnalyzing class distribution...")
    train_dist = get_class_distribution(TRAIN_DIR)
    
    print("\nTraining Set Distribution:")
    for class_name, count in train_dist.items():
        print(f"  {class_name:20s}: {count:4d} images")
    
    # Calculate class weights
    print("\nCalculating class weights (for imbalanced data)...")
    class_weights = get_class_weights(TRAIN_DIR)
    print("\nClass Weights:")
    for idx, weight in class_weights.items():
        print(f"  {CLASS_NAMES[idx]:20s}: {weight:.4f}")
    
    # Plot distribution
    print("\nPlotting class distribution...")
    plot_class_distribution(TRAIN_DIR, TEST_DIR, RESULTS_DIR + "/class_distribution.png")


def example_2_build_model():
    """
    Example 2: Build and compile model
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: MODEL BUILDING")
    print("="*70)
    
    print("\nInitializing ResNet50Classifier...")
    classifier = ResNet50Classifier(
        num_classes=NUM_CLASSES,
        learning_rate=INITIAL_LEARNING_RATE,
        img_size=IMG_SIZE,
        l2_reg=1e-4,
        dropout_rate=0.5
    )
    
    print("Building model...")
    model = classifier.build_model()
    
    print("Compiling model...")
    classifier.compile_model(model, learning_rate=INITIAL_LEARNING_RATE)
    
    print("\nModel Summary:")
    classifier.get_model_summary(model)
    
    print("\n[OK] Model ready for training!")
    
    return model, classifier


def example_3_data_loading():
    """
    Example 3: Load and prepare data with validation split
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: DATA LOADING WITH VALIDATION SPLIT")
    print("="*70)
    
    print("\nInitializing DataLoader...")
    data_loader = DataLoader(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    print("\nLoading training data with augmentation...")
    print(f"Validation split: {VALIDATION_SPLIT*100:.0f}%")
    train_gen, val_gen = data_loader.load_train_data(TRAIN_DIR, validation_split=VALIDATION_SPLIT)
    
    # Get a batch to inspect
    print("\nInspecting data batch...")
    batch_images, batch_labels = next(train_gen)
    print(f"Batch shape: {batch_images.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    print(f"Pixel value range: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
    
    return data_loader, train_gen, val_gen


def example_4_single_image_prediction():
    """
    Example 4: Predict on a single image (after training)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: SINGLE IMAGE PREDICTION")
    print("="*70)
    
    from predict import ImagePredictor
    
    try:
        print("\nInitializing ImagePredictor...")
        predictor = ImagePredictor()
        
        # Example (modify with actual image path)
        # image_path = "path/to/medical_image.jpg"
        # print(f"\nPredicting on: {image_path}")
        # result = predictor.predict_single(image_path, top_k=3)
        
        # print(f"\nPredicted Class: {result['predicted_class']}")
        # print(f"Confidence: {result['confidence']:.2%}")
        # print("\nTop-3 Predictions:")
        # for i, pred in enumerate(result['top_k_predictions'], 1):
        #     print(f"  {i}. {pred['class']:20s} {pred['probability']:.4f}")
        
        print("\n[INFO] Example prediction code:")
        print("  result = predictor.predict_single('image.jpg')")
        print("  print(result['predicted_class'])")
        
    except Exception as e:
        print(f"[INFO] Model not found - train first with: python train.py")


def example_5_evaluation_metrics():
    """
    Example 5: Calculate and visualize metrics (after training)
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: EVALUATION METRICS")
    print("="*70)
    
    try:
        print("\nInitializing Evaluator...")
        evaluator = Evaluator(class_names=CLASS_NAMES)
        
        print("\n[INFO] Example evaluation code:")
        print("  evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)")
        print("  evaluator.plot_confusion_matrix(y_true, y_pred)")
        print("  evaluator.plot_roc_auc(y_true, y_pred_proba)")
        print("  evaluator.plot_per_class_metrics(y_true, y_pred)")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")


def example_6_complete_workflow():
    """
    Example 6: Complete training workflow (pseudo-code)
    
    Note: Classes and NUM_CLASSES are automatically detected from
    subdirectory names in the training directory.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: COMPLETE WORKFLOW")
    print("="*70)
    
    print("""
# STEP 0: Configuration (Automatic)
# Classes are automatically detected from data/train subdirectory names
# NO manual class configuration needed!
from config.config import CLASS_NAMES, NUM_CLASSES
print(f'Auto-detected {NUM_CLASSES} classes: {CLASS_NAMES}')

# STEP 1: Data Preparation (with automatic validation split)
from src import DataLoader
data_loader = DataLoader(img_size=224, batch_size=32)
train_gen, val_gen = data_loader.load_train_data('data/train', validation_split=0.2)
# This splits 20% of training data for validation automatically

# STEP 2: Build Model
from src import ResNet50Classifier
classifier = ResNet50Classifier(num_classes=NUM_CLASSES, img_size=224)
model = classifier.build_model()
classifier.compile_model(model)

# STEP 3: Calculate Class Weights (optional, for imbalanced data)
from src import get_class_weights
class_weights = get_class_weights('data/train')

# STEP 4: Phase 1 - Initial Training
callbacks = classifier.get_callbacks('phase1')
history1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=20, callbacks=callbacks,
    class_weight=class_weights
)

# STEP 5: Phase 2 - Fine-tuning
classifier.unfreeze_base_model(model, from_layer=100)
classifier.compile_model(model, learning_rate=1e-5)
callbacks = classifier.get_callbacks('phase2')
history2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=10, callbacks=callbacks,
    class_weight=class_weights
)

# STEP 6: Evaluation on Test Set
from src import Evaluator
test_gen = data_loader.load_test_data('data/test')
predictions, labels = DataLoader.get_predictions_from_generator(model, test_gen)
evaluator = Evaluator(class_names=CLASS_NAMES)
metrics = evaluator.calculate_metrics(labels, predictions.argmax(1), predictions)
evaluator.plot_confusion_matrix(labels, predictions.argmax(1))
evaluator.plot_roc_auc(labels, predictions)

# STEP 7: Prediction on New Images
from predict import ImagePredictor
predictor = ImagePredictor()
result = predictor.predict_single('image.jpg')
predictor.plot_prediction('image.jpg', 'output.png')

# KEY INSIGHT: All class names (CLASS_NAMES, NUM_CLASSES) are 
# automatically detected from data/train subdirectories in alphabetical order.
# Just ensure your subdirectories are named appropriately!
    """)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MEDICAL IMAGING PROJECT - COMPLETE EXAMPLES")
    print("="*70)
    
    print("""
This script demonstrates how to use all components of the project.
Choose an example to run:

1. Data Analysis
   - Verify data structure
   - Analyze class distribution
   - Calculate class weights
   - Plot distribution
   
2. Model Building
   - Create ResNet50 classifier
   - Compile with callbacks
   - Print model summary
   
3. Data Loading
   - Load training data with augmentation
   - Load validation data
   - Inspect batch
   
4. Single Image Prediction
   - Initialize predictor
   - Predict on single image
   - Visualize predictions
   
5. Evaluation Metrics
   - Calculate metrics
   - Generate visualizations
   
6. Complete Workflow
   - Full training pipeline pseudo-code
   - From data loading to prediction

To run individual examples, uncomment in the main() section.
Or run full training with: python train.py
    """)
    
    # Uncomment to run examples
    # example_1_data_analysis()
    # example_2_build_model()
    # example_3_data_loading()
    # example_4_single_image_prediction()
    # example_5_evaluation_metrics()
    # example_6_complete_workflow()
    
    print("\nTo start training, run: python train.py")
    print("To evaluate, run: python evaluate.py")
    print("For quick setup, run: python setup.py")


if __name__ == "__main__":
    main()
