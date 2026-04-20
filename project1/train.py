"""
Main Training Script
Trains ResNet50 model with two-phase approach: initial training + fine-tuning
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config.config import (
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, INITIAL_EPOCHS, FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE, FINE_TUNE_LEARNING_RATE, VALIDATION_SPLIT ,
    TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR,
    USE_CLASS_WEIGHTS, CLASS_NAMES
)
from src.model import ResNet50Classifier
from src.data_loader import DataLoader
from src.utils import (
    get_class_weights, get_class_distribution, verify_data_structure,
    create_results_directory, save_training_history, save_predictions,
    plot_class_distribution, plot_training_history
)


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [MODELS_DIR, LOGS_DIR, RESULTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Directory ready: {directory}")


def train_model():
    """
    Main training function with two-phase approach
    
    Phase 1: Train with frozen base (ResNet50 backbone)
    Phase 2: Fine-tune with unfrozen layers
    """
    print("\n" + "="*70)
    print("MEDICAL IMAGE CLASSIFICATION - TRAINING PIPELINE")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Verify data structure
    print("\n[CONFIG] Verifying data structure...")
    verify_data_structure(TRAIN_DIR, TEST_DIR)
    
    # Get class distribution
    print("\n[ANALYSIS] Class distribution analysis:")
    train_dist = get_class_distribution(TRAIN_DIR)
    print(f"[INFO] Validation split: {VALIDATION_SPLIT*100:.0f}%")
    
    # Load data with validation split
    print("\n[DATA] Loading datasets...")
    data_loader = DataLoader(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    train_generator, val_generator = data_loader.load_train_data(
        TRAIN_DIR, VAL_DIR 
    )
    
    # Get class weights if using
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("\n[CONFIG] Calculating class weights for imbalanced data...")
        class_weights = get_class_weights(TRAIN_DIR)
        print("[OK] Class weights:")
        for cls_idx, weight in class_weights.items():
            print(f"    {CLASS_NAMES[cls_idx]:20s}: {weight:.4f}")
    
    # Initialize model
    print("\n[MODEL] Building ResNet50 model...")
    classifier = ResNet50Classifier(
        num_classes=NUM_CLASSES,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),       
        l2_reg=1e-4,
        dropout_rate=0.5
    )
    #model = classifier.build_model()
    classifier.build_model();
    classifier.compile_model(learning_rate=INITIAL_LEARNING_RATE)
    
    # Print model summary
    print("\n[MODEL] Model Architecture:")
    classifier.get_model_summary()
    
    # Phase 1: Initial Training (frozen base)
    print("\n" + "="*70)
    print("PHASE 1: INITIAL TRAINING (Frozen Base)")
    print("="*70)
    print(f"Epochs: {INITIAL_EPOCHS}")
    print(f"Learning Rate: {INITIAL_LEARNING_RATE}")
    print(f"Base Model: ResNet50 (FROZEN)")
    
    initial_checkpoint_path = os.path.join(MODELS_DIR, 'initial_training_best.h5')
    callbacks_phase1 = classifier.get_callbacks(
       model_checkpoint_path=initial_checkpoint_path,
       patience=5,
       factor=0.5,
       min_lr=1e-7
    )
    
    history_phase1 = classifier.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks_phase1,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None,
        verbose=1
    )
    
    print("[OK] Phase 1 training completed!")
    
    # Load best model from phase 1
    best_model_path = os.path.join(MODELS_DIR, "initial_training_best.h5")
    if os.path.exists(best_model_path):
        classifier.load_model(best_model_path)
        print(f"[OK] Loaded best model from: {best_model_path}")
        #classifier.model = model
    
    # Phase 2: Fine-tuning (unfrozen layers)
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNING (Unfrozen Layers)")
    print("="*70)
    print(f"Epochs: {FINE_TUNE_EPOCHS}")
    print(f"Learning Rate: {FINE_TUNE_LEARNING_RATE}")
    print(f"Base Model: ResNet50 (unfrozen from layer 100)")
    
    # Unfreeze base model
    classifier.unfreeze_base_model(from_layer=100)
    
    # Recompile with lower learning rate
    classifier.compile_model(learning_rate=FINE_TUNE_LEARNING_RATE)
    
    finetune_checkpoint_path = os.path.join(MODELS_DIR, 'finetuning_best.h5')
    callbacks_phase2 = classifier.get_callbacks(
        model_checkpoint_path=finetune_checkpoint_path,
        patience=5,
        factor=0.5,
        min_lr=1e-7
    )
    
    history_phase2 = classifier.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks_phase2,
        class_weight=class_weights if USE_CLASS_WEIGHTS else None,
        verbose=1
    )
    
    print("[OK] Phase 2 fine-tuning completed!")
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, "final_model.h5")
    classifier.save_model(final_model_path)
    print(f"[OK] Final model saved to: {final_model_path}")
    
    # Save training histories
    print("\n[RESULTS] Saving training histories...")
    history1_path = os.path.join(RESULTS_DIR, "training_history_phase1.json")
    history2_path = os.path.join(RESULTS_DIR, "training_history_phase2.json")
    
    save_training_history(history_phase1, history1_path)
    save_training_history(history_phase2, history2_path)
    
    # Plot training history
    print("[RESULTS] Plotting training history...")
    plot_training_history(history_phase1, "Phase 1 - Initial Training",
                         os.path.join(RESULTS_DIR, "training_history_phase1.png"))
    plot_training_history(history_phase2, "Phase 2 - Fine-tuning",
                         os.path.join(RESULTS_DIR, "training_history_phase2.png"))
    
    # Training summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Phase 1 Final Validation Accuracy: {history_phase1.history['val_accuracy'][-1]:.4f}")
    print(f"Phase 1 Final Validation Loss:     {history_phase1.history['val_loss'][-1]:.4f}")
    print(f"Phase 2 Final Validation Accuracy: {history_phase2.history['val_accuracy'][-1]:.4f}")
    print(f"Phase 2 Final Validation Loss:     {history_phase2.history['val_loss'][-1]:.4f}")
    print(f"\nModel saved to: {final_model_path}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)
    
    # Save training info
    training_info = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "epochs_phase1": INITIAL_EPOCHS,
            "epochs_phase2": FINE_TUNE_EPOCHS,
            "learning_rate_phase1": INITIAL_LEARNING_RATE,
            "learning_rate_phase2": FINE_TUNE_LEARNING_RATE,
            "use_class_weights": USE_CLASS_WEIGHTS,
        },
        "data": {
            "train_samples": train_generator.n,
            "val_samples": val_generator.n,
            "classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        },
        "results": {
            "phase1_val_accuracy": float(history_phase1.history['val_accuracy'][-1]),
            "phase1_val_loss": float(history_phase1.history['val_loss'][-1]),
            "phase2_val_accuracy": float(history_phase2.history['val_accuracy'][-1]),
            "phase2_val_loss": float(history_phase2.history['val_loss'][-1]),
        }
    }
    
    info_path = os.path.join(RESULTS_DIR, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"[OK] Training info saved to: {info_path}")
    
    return classifier.model, train_generator, val_generator


if __name__ == "__main__":
    try:
        model, train_gen, val_gen = train_model()
        print("\n[OK] Training pipeline completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
