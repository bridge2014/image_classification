"""
Example script for making predictions on single medical images
After training, use this to make predictions on new images
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config


def predict_single_image(image_path, model_path=config.FINAL_MODEL_PATH):
    """
    Make prediction on a single medical image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB')
    
    # Resize to model input size
    img = img.resize(config.IMAGE_SIZE)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize (same as training/evaluation)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get all class probabilities
    all_probs = predictions[0]
    
    return predicted_class, confidence, all_probs


def predict_and_display(image_path, model_path=config.FINAL_MODEL_PATH):
    """
    Make prediction and display results
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
    """
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    # Make prediction
    predicted_class, confidence, all_probs = predict_single_image(image_path, model_path)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    print("\nClass Probabilities (Top 5):")
    print("-" * 60)
    
    # Get top 5 predictions
    top_5_indices = np.argsort(all_probs)[-5:][::-1]
    
    for rank, idx in enumerate(top_5_indices, 1):
        prob = all_probs[idx]
        print(f"{rank}. Class {idx}: {prob:.4f} ({prob*100:.2f}%)")
    
    print("="*60 + "\n")


def predict_batch(image_dir, model_path=config.FINAL_MODEL_PATH):
    """
    Make predictions on multiple images in a directory
    
    Args:
        image_dir: Directory containing images
        model_path: Path to the trained model
    """
    
    if not os.path.isdir(image_dir):
        print(f"ERROR: Directory not found: {image_dir}")
        return
    
    # Load model once
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"ERROR: No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(config.IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            results.append({
                'filename': image_file,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            print(f"✓ {image_file:30} → Class {predicted_class} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"✗ {image_file:30} → ERROR: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    
    # Group by class
    class_counts = {}
    for result in results:
        cls = result['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("\nPredictions by class:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        percentage = (count / len(results)) * 100
        print(f"  Class {cls}: {count} images ({percentage:.1f}%)")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    print("="*60 + "\n")


def main():
    """Main function with command line interface"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Make predictions using trained medical image classifier"
    )
    parser.add_argument(
        "-i", "--image",
        help="Path to single image for prediction"
    )
    parser.add_argument(
        "-d", "--directory",
        help="Directory containing images for batch prediction"
    )
    parser.add_argument(
        "-m", "--model",
        default=config.FINAL_MODEL_PATH,
        help=f"Path to trained model (default: {config.FINAL_MODEL_PATH})"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        print("Please train the model first using: python train_model.py")
        return
    
    # Single image prediction
    if args.image:
        predict_and_display(args.image, args.model)
    
    # Batch prediction
    elif args.directory:
        predict_batch(args.directory, args.model)
    
    else:
        # Example usage if no arguments provided
        print("Medical Image Classification - Prediction Script")
        print("=" * 60)
        print("\nUsage:")
        print("  Single image:  python predict.py -i path/to/image.jpg")
        print("  Batch images:  python predict.py -d path/to/directory")
        print("  Custom model:  python predict.py -i image.jpg -m model.h5")
        print("\nExample:")
        print("  python predict.py -i data/test/class_0/image_001.jpg")
        print("  python predict.py -d data/test/class_0")
        print("\nNote:")
        print("  - Model will be loaded from config.FINAL_MODEL_PATH")
        print("  - Images are resized to 224x224")
        print("  - Output shows predicted class and confidence")
        print("=" * 60)


if __name__ == "__main__":
    main()
