"""
Main script to evaluate the trained medical image classification model
Run this script to evaluate on test dataset and generate metrics/visualizations
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluate import ModelEvaluator
from src.utils import verify_data_structure, plot_training_history
import src.config as config


def main():
    """
    Main evaluation entry point
    """
    print("\n" + "="*80)
    print("MEDICAL IMAGE CLASSIFICATION - MODEL EVALUATION")
    print("="*80)

    # Check if model exists
    if not os.path.exists(config.FINAL_MODEL_PATH):
        print(f"\n[ERROR] Model not found at {config.FINAL_MODEL_PATH}")
        print("Please train the model first using: python train_model.py")
        sys.exit(1)

    # Verify test data exists
    print("\nVerifying data structure...")
    verify_data_structure()

    print("\nStarting evaluation...")

    # Evaluate model
    try:
        evaluator = ModelEvaluator(model_path=config.FINAL_MODEL_PATH)
        metrics = evaluator.evaluate()

        print("\n[SUCCESS] Evaluation completed!")
        print(f"\nResults saved to: {config.RESULTS_DIR}")
        print("\nGenerated files:")
        print(f"  - Confusion Matrix: {config.CONFUSION_MATRIX_PATH}")
        print(f"  - ROC/AUC Curves: {config.ROC_CURVE_PATH}")
        print(f"  - Classification Report: {config.CLASSIFICATION_REPORT_PATH}")
        print(f"  - Metrics: {config.METRICS_PATH}")

        # Try to plot training history if available
        if os.path.exists(config.TRAINING_HISTORY_PATH):
            print("\nGenerating training history plots...")
            try:
                plot_training_history()
            except Exception as e:
                print(f"Could not plot training history: {e}")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
