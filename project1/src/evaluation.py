"""
Evaluation Module
Metrics calculation, visualization, and reporting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd


class Evaluator:
    """
    Comprehensive evaluation metrics and visualization
    """
    
    def __init__(self, class_names, dpi=100):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names
            dpi: DPI for plot figures
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.dpi = dpi
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels (class indices)
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        metrics['per_class'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC for binary classification or one-vs-rest multiclass
        if self.num_classes == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba is not None else y_pred)
            except:
                metrics['roc_auc'] = None
        else:
            try:
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(
                        label_binarize(y_true, classes=range(self.num_classes)),
                        y_pred_proba,
                        multi_class='ovr',
                        average='weighted'
                    )
                else:
                    metrics['roc_auc'] = None
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics in formatted way"""
        print("\n" + "="*70)
        print("EVALUATION METRICS")
        print("="*70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("="*70)
        
        print("\nCLASSIFICATION REPORT:")
        print("="*70)
        print(classification_report(
            metrics['per_class'],
            target_names=self.class_names,
            zero_division=0
        ))
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path="confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Where to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
            square=True,
            annot_kws={'size': 10}
        )
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to: {output_path}")
        plt.close()
    
    def plot_roc_auc(self, y_true, y_pred_proba, output_path="roc_auc.png"):
        """
        Plot ROC-AUC curves
        
        Args:
            y_true: True labels (one-hot encoded or class indices)
            y_pred_proba: Prediction probabilities for all classes
            output_path: Where to save plot
        """
        # Convert y_true if needed
        if len(y_true.shape) == 1:
            y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
        else:
            y_true_binarized = y_true
        
        plt.figure(figsize=(12, 8))
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for All Classes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"[OK] ROC-AUC plot saved to: {output_path}")
        plt.close()
    
    def plot_per_class_metrics(self, y_true, y_pred, output_path="per_class_metrics.png"):
        """
        Plot per-class precision, recall, and F1-score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Where to save plot
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=None,
            zero_division=0
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Precision
        axes[0].bar(self.class_names, precision, color='steelblue', edgecolor='black')
        axes[0].set_title('Precision by Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Precision')
        axes[0].set_ylim([0, 1.1])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Recall
        axes[1].bar(self.class_names, recall, color='coral', edgecolor='black')
        axes[1].set_title('Recall by Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Recall')
        axes[1].set_ylim([0, 1.1])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # F1-Score
        axes[2].bar(self.class_names, f1, color='lightgreen', edgecolor='black')
        axes[2].set_title('F1-Score by Class', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_ylim([0, 1.1])
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"[OK] Per-class metrics plot saved to: {output_path}")
        plt.close()
    
    def plot_training_comparison(self, history1, history2, output_path="training_comparison.png"):
        """
        Plot comparison of initial and fine-tuned training
        
        Args:
            history1: Initial training history
            history2: Fine-tuned training history
            output_path: Where to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Combine histories
        epochs1 = len(history1.history['loss'])
        epochs2 = len(history2.history['loss'])
        
        combined_loss = history1.history['loss'] + history2.history['loss']
        combined_acc = history1.history['accuracy'] + history2.history['accuracy']
        combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']
        combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        
        all_epochs = list(range(1, len(combined_loss) + 1))
        
        # Loss
        axes[0].plot(all_epochs[:epochs1], history1.history['loss'], 'b-', label='Train (Phase 1)', linewidth=2)
        axes[0].plot(all_epochs[:epochs1], history1.history['val_loss'], 'b--', label='Val (Phase 1)', linewidth=2)
        axes[0].plot(all_epochs[epochs1:], history2.history['loss'], 'r-', label='Train (Phase 2)', linewidth=2)
        axes[0].plot(all_epochs[epochs1:], history2.history['val_loss'], 'r--', label='Val (Phase 2)', linewidth=2)
        axes[0].axvline(x=epochs1, color='green', linestyle=':', linewidth=2, label='Fine-tuning Start')
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(all_epochs[:epochs1], history1.history['accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
        axes[1].plot(all_epochs[:epochs1], history1.history['val_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
        axes[1].plot(all_epochs[epochs1:], history2.history['accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
        axes[1].plot(all_epochs[epochs1:], history2.history['val_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
        axes[1].axvline(x=epochs1, color='green', linestyle=':', linewidth=2, label='Fine-tuning Start')
        axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        print(f"[OK] Training comparison plot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
