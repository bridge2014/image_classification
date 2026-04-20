"""
Data loading and augmentation utilities for medical image classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import config


class DataLoader:
    """
    Handles data loading, augmentation, and preprocessing for medical images
    """

    def __init__(self, image_size=config.IMAGE_SIZE, batch_size=config.BATCH_SIZE):
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.class_weights = None
        self.class_indices = None

    def get_augmentation_generator(self):
        """
        Create data augmentation generator for training data
        
        Returns:
            ImageDataGenerator: Configured data augmentation generator
        """
        return ImageDataGenerator(
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            brightness_range=config.BRIGHTNESS_RANGE,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            vertical_flip=config.VERTICAL_FLIP,
            fill_mode=config.FILL_MODE,
            rescale=1./255
        )

    def get_normalization_generator(self):
        """
        Create normalization generator for validation and test data (no augmentation)
        
        Returns:
            ImageDataGenerator: Configured normalization generator
        """
        return ImageDataGenerator(rescale=1./255)

    def load_data(self):
        """
        Load training, validation, and test datasets from directories
        
        Returns:
            tuple: (train_generator, val_generator, test_generator, class_weights)
        """
        print("Loading data from directories...")
        
        # Create augmentation generators
        train_augmentation = self.get_augmentation_generator()
        val_test_normalization = self.get_normalization_generator()

        # Load training data with augmentation
        print(f"Loading training data from {config.TRAIN_DIR}")
        self.train_generator = train_augmentation.flow_from_directory(
            config.TRAIN_DIR,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            seed=config.RANDOM_SEED
        )

        # Load validation data without augmentation
        print(f"Loading validation data from {config.VAL_DIR}")
        self.val_generator = val_test_normalization.flow_from_directory(
            config.VAL_DIR,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            seed=config.RANDOM_SEED
        )

        # Load test data without augmentation
        print(f"Loading test data from {config.TEST_DIR}")
        self.test_generator = val_test_normalization.flow_from_directory(
            config.TEST_DIR,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=config.RANDOM_SEED
        )

        # Store class indices for later use
        self.class_indices = self.train_generator.class_indices

        # Calculate class weights to handle imbalanced data
        self.class_weights = self._calculate_class_weights()

        print("Data loading completed!")
        print(f"Number of training samples: {self.train_generator.samples}")
        print(f"Number of validation samples: {self.val_generator.samples}")
        print(f"Number of test samples: {self.test_generator.samples}")
        print(f"Classes: {list(self.class_indices.keys())}")
        print(f"Class weights: {self.class_weights}")

        return self.train_generator, self.val_generator, self.test_generator, self.class_weights

    def _calculate_class_weights(self):
        """
        Calculate class weights to handle imbalanced data
        
        Returns:
            dict: Class weights dictionary
        """
        print("\nCalculating class weights for imbalanced data...")
        
        class_weights_dict = {}
        
        # Count samples per class in training directory
        class_counts = {}
        for class_name in os.listdir(config.TRAIN_DIR):
            class_path = os.path.join(config.TRAIN_DIR, class_name)
            if os.path.isdir(class_path):
                num_samples = len(os.listdir(class_path))
                class_counts[class_name] = num_samples
        
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        # Calculate weights (higher weight for minority classes)
        for idx, (class_name, count) in enumerate(class_counts.items()):
            weight = (total_samples) / (num_classes * count)
            class_weights_dict[idx] = weight
            print(f"Class {class_name}: {count} samples -> weight: {weight:.4f}")
        
        return class_weights_dict

    def get_generators(self):
        """Get the current generators"""
        return self.train_generator, self.val_generator, self.test_generator

    def get_class_weights(self):
        """Get class weights"""
        return self.class_weights

    def get_class_indices(self):
        """Get class indices mapping"""
        return self.class_indices


def create_sample_data_structure():
    """
    Create sample directory structure for demonstration purposes
    This helps users understand the expected data format
    """
    import shutil
    from PIL import Image
    
    print("Creating sample data structure...")
    
    # Create sample directories
    for split in ['train', 'val', 'test']:
        for class_idx in range(config.NUM_CLASSES):
            class_dir = os.path.join(config.DATA_DIR, split, f'class_{class_idx}')
            os.makedirs(class_dir, exist_ok=True)
            
            # Create dummy images for demonstration
            if len(os.listdir(class_dir)) == 0:
                for img_idx in range(5):  # 5 images per class for demo
                    # Create a simple RGB image
                    img = Image.new('RGB', (224, 224), 
                                  color=(np.random.randint(0, 256), 
                                        np.random.randint(0, 256), 
                                        np.random.randint(0, 256)))
                    img_path = os.path.join(class_dir, f'image_{img_idx}.jpg')
                    img.save(img_path)
    
    print("Sample data structure created!")
