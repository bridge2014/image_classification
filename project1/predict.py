"""
Prediction Script
Inference on new medical images
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from config.config import IMG_SIZE, NUM_CLASSES, MODELS_DIR, CLASS_NAMES
from src.model import ResNet50Classifier


class ImagePredictor:
    """
    Predicts class labels for new medical images
    """
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model (if None, uses default)
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "final_model.h5")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.classifier = ResNet50Classifier(
            num_classes=NUM_CLASSES,
            img_size=IMG_SIZE
        )
        self.model = self.classifier.load_model(model_path)
        print(f"[OK] Model loaded from: {model_path}")
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, img_array
    
    def predict_single(self, image_path, top_k=3):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
        
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess
        img_batch, img_array = self.load_and_preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = predictions[top_indices]
        top_classes = [CLASS_NAMES[idx] for idx in top_indices]
        
        result = {
            'image_path': image_path,
            'predicted_class': CLASS_NAMES[np.argmax(predictions)],
            'confidence': float(predictions.max()),
            'top_k_predictions': [
                {
                    'class': top_classes[i],
                    'probability': float(top_probs[i])
                }
                for i in range(len(top_classes))
            ],
            'all_probabilities': {
                CLASS_NAMES[i]: float(predictions[i])
                for i in range(len(CLASS_NAMES))
            }
        }
        
        return result
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            top_k: Return top-k predictions
        
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, top_k=top_k)
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to predict for {image_path}: {str(e)}")
        
        return results
    
    def plot_prediction(self, image_path, output_path=None):
        """
        Plot image with prediction
        
        Args:
            image_path: Path to image file
            output_path: Where to save plot (if None, displays plot)
        """
        # Get prediction
        result = self.predict_single(image_path, top_k=3)
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Medical Image', fontsize=12, fontweight='bold')
        
        # Display predictions
        classes = [p['class'] for p in result['top_k_predictions']]
        probs = [p['probability'] for p in result['top_k_predictions']]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax2.barh(range(len(classes)), probs, color=colors[:len(classes)], edgecolor='black')
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Probability', fontsize=11)
        ax2.set_xlim([0, 1.0])
        ax2.set_title('Top-3 Predictions', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, prob in enumerate(probs):
            ax2.text(prob + 0.01, i, f'{prob:.2%}', va='center', fontsize=10)
        
        plt.suptitle(
            f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})",
            fontsize=13, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            print(f"[OK] Prediction plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Example usage"""
    print("\n" + "="*70)
    print("IMAGE PREDICTION")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = ImagePredictor()
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        print("[INFO] Please train the model first using train.py")
        return
    
    # Example: Predict on a single image
    # Uncomment and modify the image_path
    # result = predictor.predict_single("path/to/image.jpg")
    # print("\n[RESULT] Single Image Prediction:")
    # print(f"Predicted Class: {result['predicted_class']}")
    # print(f"Confidence: {result['confidence']:.4f}")
    # print(f"\nTop-3 Predictions:")
    # for i, pred in enumerate(result['top_k_predictions'], 1):
    #     print(f"  {i}. {pred['class']:20s} {pred['probability']:.4f}")
    
    print("\n[INFO] To use the predictor:")
    print("from predict import ImagePredictor")
    print("predictor = ImagePredictor()")
    print("result = predictor.predict_single('path/to/image.jpg')")
    print("predictor.plot_prediction('path/to/image.jpg', 'output.png')")


if __name__ == "__main__":
    main()
