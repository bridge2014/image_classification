"""
Evaluation script for medical image classification model
Generates classification reports, confusion matrix, and ROC/AUC plots
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import pickle
import os
from data_loader import DataLoader
import config


class ModelEvaluator:
    """
    Evaluates model performance on test dataset
    Generates comprehensive evaluation metrics and visualizations
    """

    def __init__(self, model_path=config.FINAL_MODEL_PATH):
        self.model = None
        self.model_path = model_path
        self.data_loader = DataLoader()
        self.class_indices = None
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None

    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return self.model

    def evaluate(self):
        """
        Complete evaluation pipeline:
        1. Load model and data
        2. Make predictions
        3. Generate evaluation metrics
        4. Create visualizations
        """
        print("="*80)
        print("MODEL EVALUATION")
        print("="*80)

        # Load model
        print("\n[STEP 1] Loading Model...")
        self.load_model()

        # Load data
        print("\n[STEP 2] Loading Test Data...")
        train_gen, val_gen, test_gen, class_weights = self.data_loader.load_data()
        self.class_indices = self.data_loader.get_class_indices()

        # Get predictions
        print("\n[STEP 3] Generating Predictions...")
        self.y_pred_proba = self.model.predict(test_gen)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        self.y_true = test_gen.classes

        # Generate metrics
        print("\n[STEP 4] Calculating Metrics...")
        metrics = self._calculate_metrics()

        # Save classification report
        print("\n[STEP 5] Generating Classification Report...")
        self._save_classification_report()

        # Create visualizations
        print("\n[STEP 6] Creating Visualizations...")
        self._plot_confusion_matrix()
        self._plot_roc_curves()

        # Save metrics
        self._save_metrics(metrics)

        print("\n" + "="*80)
        print("EVALUATION COMPLETED")
        print("="*80)

        return metrics

    def _calculate_metrics(self):
        """
        Calculate evaluation metrics
        
        Returns:
            dict: Dictionary containing various metrics
        """
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'macro_precision': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
        }

        # Calculate ROC-AUC
        try:
            y_true_bin = label_binarize(self.y_true, classes=range(config.NUM_CLASSES))
            metrics['roc_auc_macro'] = roc_auc_score(y_true_bin, self.y_pred_proba, average='macro', multi_class='ovr')
            metrics['roc_auc_weighted'] = roc_auc_score(y_true_bin, self.y_pred_proba, average='weighted', multi_class='ovr')
        except:
            metrics['roc_auc_macro'] = 0.0
            metrics['roc_auc_weighted'] = 0.0

        print("\nMetrics Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score: {metrics['macro_f1']:.4f}")
        print(f"  ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        print(f"  ROC-AUC (Weighted): {metrics['roc_auc_weighted']:.4f}")

        return metrics

    def _save_classification_report(self):
        """Save detailed classification report"""
        # Create reverse mapping from indices to class names
        idx_to_class = {v: k for k, v in self.class_indices.items()}
        target_names = [idx_to_class[i] for i in range(config.NUM_CLASSES)]

        report = classification_report(
            self.y_true, self.y_pred,
            target_names=target_names,
            digits=4
        )

        print("\nClassification Report:")
        print(report)

        # Save to file
        with open(config.CLASSIFICATION_REPORT_PATH, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(report)
            f.write("\n" + "="*80 + "\n")

        print(f"Classification report saved to {config.CLASSIFICATION_REPORT_PATH}")

    def _plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=range(config.NUM_CLASSES),
            yticklabels=range(config.NUM_CLASSES)
        )
        plt.title('Confusion Matrix - Test Dataset', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_PATH}")
        plt.close()

    def _plot_roc_curves(self):
        """Plot and save ROC curves for all classes"""
        # Binarize the labels for multi-class ROC
        y_true_bin = label_binarize(self.y_true, classes=range(config.NUM_CLASSES))

        # Create figure with subplots for ROC curves
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()

        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        # Plot ROC curve for each class
        for i in range(config.NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            roc_auc_dict[i] = roc_auc

            axes[i].plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - Class {i}')
            axes[i].legend(loc="lower right")
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('ROC Curves for All Classes', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(config.ROC_CURVE_PATH, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {config.ROC_CURVE_PATH}")
        plt.close()

        # Print AUC scores
        print("\nPer-Class ROC-AUC Scores:")
        for i, auc_score in roc_auc_dict.items():
            print(f"  Class {i}: {auc_score:.4f}")

    def _save_metrics(self, metrics):
        """Save metrics to text file"""
        with open(config.METRICS_PATH, 'w') as f:
            f.write("MODEL EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")
            
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Results saved to: {config.RESULTS_DIR}\n")

        print(f"Metrics saved to {config.METRICS_PATH}")


def main():
    """Main evaluation function"""
    try:
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate()

        print("\n[SUCCESS] Evaluation completed successfully!")
        print(f"Results saved to: {config.RESULTS_DIR}")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
