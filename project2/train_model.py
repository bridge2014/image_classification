"""
Main script to train the medical image classification model
Run this script to start training with data augmentation, class weights, and fine-tuning
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import ModelTrainer
from src.data_loader import create_sample_data_structure
from src.utils import print_project_structure, verify_data_structure
import src.config as config


def main():
    """
    Main training entry point
    """
    print("\n" + "="*80)
    print("MEDICAL IMAGE CLASSIFICATION - TRAINING INITIALIZATION")
    print("="*80)

    # Show project structure
    print_project_structure()

    # Verify data structure
    verify_data_structure()

    # Ask user to confirm data is ready
    response = input("\n\nHave you placed your training/validation/test data? (yes/no): ").strip().lower()

    if response == 'no':
        print("\nPlease organize your data in the following structure:")
        print("  data/train/class_0/, class_1/, ..., class_9/")
        print("  data/val/class_0/, class_1/, ..., class_9/")
        print("  data/test/class_0/, class_1/, ..., class_9/")
        print("\nYou can create sample data structure with:")
        print("  from src.data_loader import create_sample_data_structure")
        print("  create_sample_data_structure()")
        return

    print("\nStarting training process...")
    print("\nConfiguration:")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Image Size: {config.IMAGE_SIZE}")
    print(f"  - Initial Training Epochs (Head): {config.INITIAL_EPOCHS}")
    print(f"  - Fine-tuning Epochs: {config.FINE_TUNE_EPOCHS}")
    print(f"  - Learning Rate (Initial): {config.LEARNING_RATE}")
    print(f"  - Learning Rate (Fine-tune): {config.FINE_TUNE_LEARNING_RATE}")
    print(f"  - Data Augmentation: Enabled")
    print(f"  - Class Weights: Enabled (for imbalanced data)")

    # Start training
    try:
        trainer = ModelTrainer()
        model, history = trainer.train()
        print("\n[SUCCESS] Training completed!")
        print(f"Trained model saved at: {config.FINAL_MODEL_PATH}")
        print(f"Best model saved at: {config.BEST_MODEL_PATH}")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
