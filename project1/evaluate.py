"""
Evaluation Script
Evaluates trained model on test set with comprehensive metrics and visualizations
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.metrics import classification_report
from config.config import (
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, TEST_DIR, MODELS_DIR,
    RESULTS_DIR, CLASS_NAMES
)
from src.model import ResNet50Classifier
from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.utils import create_results_directory, save_predictions


def evaluate_model():
    """
    Evaluate trained model on test set
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Load test data
    print("\n[DATA] Loading test dataset...")
    data_loader = DataLoader(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    test_generator = data_loader.load_test_data(TEST_DIR)
    
    # Load trained model
    print("\n[MODEL] Loading trained model...")
    classifier = ResNet50Classifier(
        num_classes=NUM_CLASSES
    )
    
    model_path = os.path.join(MODELS_DIR, "final_model.h5")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at: {model_path}")
        print("[INFO] Please run train.py first to train the model")
        return
    
    model = classifier.load_model(model_path)
    print(f"[OK] Model loaded from: {model_path}")
    
    # Get predictions
    print("\n[EVAL] Generating predictions on test set...")
    test_generator.reset()
    predictions_proba, true_labels = DataLoader.get_predictions_from_generator(
        model, test_generator, num_samples=test_generator.n
    )
    
    # Convert probabilities to class predictions
    predicted_classes = np.argmax(predictions_proba, axis=1)
    
    print(f"[OK] Predictions generated for {len(true_labels)} samples")
    
    # Initialize evaluator
    evaluator = Evaluator(class_names=CLASS_NAMES, dpi=100)
    
    # Calculate metrics
    print("\n[METRICS] Calculating evaluation metrics...")
    metrics = evaluator.calculate_metrics(
        true_labels, predicted_classes, predictions_proba
    )
    
    # Print metrics
    #evaluator.print_metrics(metrics)
    
    # Create evaluation results directory
    eval_dir = create_results_directory(RESULTS_DIR)
    
    # Plot confusion matrix
    print("\n[PLOTS] Generating confusion matrix...")
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    evaluator.plot_confusion_matrix(true_labels, predicted_classes, cm_path)
    
    # Plot ROC-AUC curves
    print("[PLOTS] Generating ROC-AUC curves...")
    roc_path = os.path.join(eval_dir, "roc_auc_curves.png")
    evaluator.plot_roc_auc(true_labels, predictions_proba, roc_path)
    
    # Plot per-class metrics
    print("[PLOTS] Generating per-class metrics...")
    metrics_path = os.path.join(eval_dir, "per_class_metrics.png")
    evaluator.plot_per_class_metrics(true_labels, predicted_classes, metrics_path)
    
    # Save detailed classification report
    print("\n[RESULTS] Saving detailed classification report...")
    report = classification_report(
        true_labels, predicted_classes,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    
    report_path = os.path.join(eval_dir, "classification_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Classification report saved to: {report_path}")
    
    # Save predictions
    print("[RESULTS] Saving predictions...")
    pred_csv_path = os.path.join(eval_dir, "predictions.csv")
    save_predictions(
        true_labels, predicted_classes, predictions_proba,
        CLASS_NAMES, pred_csv_path
    )
    
    # Save evaluation summary
    print("[RESULTS] Saving evaluation summary...")
    summary = {
        "test_samples": len(true_labels),
        "accuracy": float(metrics['accuracy']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "f1_score": float(metrics['f1']),
        "roc_auc": float(metrics['roc_auc']) if metrics['roc_auc'] is not None else None,
        "per_class_metrics": metrics['per_class']
    }
    
    summary_path = os.path.join(eval_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Evaluation summary saved to: {summary_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Test Samples: {len(true_labels)}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    print("\nResults saved to:")
    print(f"  Directory: {eval_dir}")
    print(f"  - Confusion Matrix: confusion_matrix.png")
    print(f"  - ROC-AUC Curves: roc_auc_curves.png")
    print(f"  - Per-Class Metrics: per_class_metrics.png")
    print(f"  - Classification Report: classification_report.json")
    print(f"  - Predictions: predictions.csv")
    print(f"  - Summary: evaluation_summary.json")
    print("="*70)


if __name__ == "__main__":
    try:
        evaluate_model()
        print("\n[OK] Evaluation completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
