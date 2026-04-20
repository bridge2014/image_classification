"""
Training script for medical image classification model
Includes data augmentation, class weights for imbalance, and fine-tuning
"""

import tensorflow as tf
import numpy as np
import pickle
import os
from data_loader import DataLoader
from model import MedicalImageClassifier
import config


class ModelTrainer:
    """
    Handles model training, validation, and fine-tuning
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.model_classifier = MedicalImageClassifier()
        self.history = None

    def train(self):
        """
        Complete training pipeline:
        1. Load and prepare data
        2. Build and compile model
        3. Train head layers
        4. Fine-tune base model
        """
        print("="*80)
        print("MEDICAL IMAGE CLASSIFICATION - TRAINING PIPELINE")
        print("="*80)

        # Step 1: Load data
        print("\n[STEP 1] Loading Data...")
        train_gen, val_gen, test_gen, class_weights = self.data_loader.load_data()

        # Step 2: Build model
        print("\n[STEP 2] Building Model...")
        self.model_classifier.build_model()
        self.model_classifier.compile_model(learning_rate=config.LEARNING_RATE)
        self.model_classifier.summary()

        # Step 3: Train head layers (top layers) first
        print("\n[STEP 3] Training Head Layers...")
        print(f"Training for {config.INITIAL_EPOCHS} epochs with frozen base model")

        history_head = self.model_classifier.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=self.model_classifier.get_callbacks(),
            verbose=1
        )

        # Step 4: Fine-tune base model
        print("\n[STEP 4] Fine-tuning Base Model...")
        print(f"Unfreezing base model layers for fine-tuning...")

        self.model_classifier.unfreeze_base_model(num_layers_to_unfreeze=50)

        # Recompile with lower learning rate for fine-tuning
        self.model_classifier.compile_model(learning_rate=config.FINE_TUNE_LEARNING_RATE)

        print(f"\nFine-tuning for {config.FINE_TUNE_EPOCHS} epochs...")

        history_finetune = self.model_classifier.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.FINE_TUNE_EPOCHS,
            initial_epoch=config.INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=self.model_classifier.get_callbacks(),
            verbose=1
        )

        # Combine histories
        self.history = self._combine_histories(history_head, history_finetune)

        # Step 5: Save final model
        print("\n[STEP 5] Saving Model...")
        self.model_classifier.model.save(config.FINAL_MODEL_PATH)
        print(f"Model saved to {config.FINAL_MODEL_PATH}")

        # Save training history
        with open(config.TRAINING_HISTORY_PATH, 'wb') as f:
            pickle.dump(self.history, f)
        print(f"Training history saved to {config.TRAINING_HISTORY_PATH}")

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)

        return self.model_classifier.model, self.history

    def _combine_histories(self, hist1, hist2):
        """
        Combine histories from different training phases
        
        Args:
            hist1: History from first phase (head training)
            hist2: History from second phase (fine-tuning)
            
        Returns:
            dict: Combined history
        """
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        return combined


def main():
    """Main training function"""
    try:
        trainer = ModelTrainer()
        model, history = trainer.train()

        print("\n[SUCCESS] Training pipeline completed successfully!")
        print(f"Best model saved: {config.BEST_MODEL_PATH}")
        print(f"Final model saved: {config.FINAL_MODEL_PATH}")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
