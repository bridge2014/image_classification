"""
Data Loading Module
ImageDataGenerator setup with augmentation and batch processing
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import (
    IMG_SIZE,
    BATCH_SIZE,
    AUGMENTATION_CONFIG
)


class DataLoader:
    """
    Manages data loading with augmentation for train/val/test datasets
    """
    
    def __init__(self, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
        """
        Initialize DataLoader
        
        Args:
            img_size: Image size (width, height)
            batch_size: Batch size for generators
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
    
    def create_augmentation_generator(self, augmentation_config):
        """
        Create ImageDataGenerator with augmentation parameters
        
        Args:
            augmentation_config: Dictionary with augmentation parameters
        
        Returns:
            ImageDataGenerator instance
        """
        # In data_loader.py, enhance augmentation
        AUGMENTATION_CONFIG = {
            'rotation_range': 30,
            'width_shift_range': 0.3,
            'height_shift_range': 0.3,
            'shear_range': 0.3,
            'zoom_range': 0.3,
            'horizontal_flip': True,
            'vertical_flip': False,  # Medical images often shouldn't be vertically flipped
            'brightness_range': [0.8, 1.2],  # Add brightness variation
            'fill_mode': 'reflect'  # Better than 'nearest' for medical images
        }

        generator = ImageDataGenerator(
            rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
            rotation_range=augmentation_config.get('rotation', 20),
            width_shift_range=augmentation_config.get('width_shift', 0.2),
            height_shift_range=augmentation_config.get('height_shift', 0.2),
            shear_range=augmentation_config.get('shear', 0.2),
            zoom_range=augmentation_config.get('zoom', 0.2),
            horizontal_flip=augmentation_config.get('horizontal_flip', True),
            vertical_flip=augmentation_config.get('vertical_flip', False),
            fill_mode='nearest'
        )
        return generator
    
    def create_validation_generator(self):
        """
        Create ImageDataGenerator for validation (no augmentation, only normalization)
        
        Returns:
            ImageDataGenerator instance
        """
        generator = ImageDataGenerator(rescale=1.0 / 255.0)
        return generator
    
    def load_train_data(self, train_dir, val_dir , class_mode='categorical'):
        """
        Load training data with augmentation and create validation split
        
        Args:
            train_dir: Path to training directory
            val_dir: Path to validating directory
            class_mode: 'categorical' or 'binary'
            validation_split: Fraction of data to use for validation (default 0.2 = 20%)
        
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        # Create augmentation generator for training data
        aug_generator = self.create_augmentation_generator(AUGMENTATION_CONFIG)
        
        # Create validation generator (no augmentation)
        val_generator_obj = self.create_validation_generator()
        
        # Load training data with augmentation
        self.train_generator = aug_generator.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=True,
            seed=42,            
        )
        
        # Load validation data (same directory, but validation subset)
        self.val_generator = val_generator_obj.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=42,            
        )
        
        total_samples = self.train_generator.n + self.val_generator.n
        print(f"[OK] Training data loaded from: {train_dir}")
        print(f"    - Total samples: {total_samples}")
        print(f"    - Training samples: {self.train_generator.n} ({(1-0.2)*100:.0f}%)")
        print(f"    - Validation samples: {self.val_generator.n} ({0.2*100:.0f}%)")
        print(f"    - Batch size: {self.batch_size}")
        print(f"    - Training steps per epoch: {len(self.train_generator)}")
        print(f"    - Validation steps: {len(self.val_generator)}")
        
        return self.train_generator, self.val_generator
    
    def load_validation_data(self, val_dir, class_mode='categorical'):
        """
        DEPRECATED: Use load_train_data with validation_split parameter instead
        Load validation data (no augmentation)
        
        Args:
            val_dir: Path to validation directory
            class_mode: 'categorical' or 'binary'
        
        Returns:
            Validation generator
        """
        print("[WARNING] load_validation_data is deprecated!")
        print("[INFO] Use load_train_data with validation_split parameter instead")
        print("[INFO] Example: train_gen, val_gen = data_loader.load_train_data(train_dir, validation_split=0.2)")
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        val_generator = self.create_validation_generator()
        
        self.val_generator = val_generator.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=42
        )
        
        print(f"[OK] Validation data loaded from: {val_dir}")
        print(f"    - Total samples: {self.val_generator.n}")
        print(f"    - Batch size: {self.batch_size}")
        print(f"    - Steps per epoch: {len(self.val_generator)}")
        
        return self.val_generator
    
    def load_test_data(self, test_dir, class_mode='categorical'):
        """
        Load test data (no augmentation)
        
        Args:
            test_dir: Path to test directory
            class_mode: 'categorical' or 'binary'
        
        Returns:
            Test generator
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        test_generator = self.create_validation_generator()
        
        self.test_generator = test_generator.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=False,
            seed=42
        )
        
        print(f"[OK] Test data loaded from: {test_dir}")
        print(f"    - Total samples: {self.test_generator.n}")
        print(f"    - Batch size: {self.batch_size}")
        print(f"    - Steps per epoch: {len(self.test_generator)}")
        
        return self.test_generator
    
    def get_class_mapping(self):
        """
        Get class index to class name mapping
        
        Returns:
            Dictionary mapping class indices to class names
        """
        if self.train_generator is None:
            raise ValueError("Train generator not loaded yet")
        
        return self.train_generator.class_indices
    
    def get_num_classes(self):
        """
        Get number of classes
        
        Returns:
            Number of classes
        """
        if self.train_generator is None:
            raise ValueError("Train generator not loaded yet")
        
        return self.train_generator.num_classes
    
    @staticmethod
    def get_predictions_from_generator(model, generator, num_samples=None):
        """
        Get predictions from a generator
        
        Args:
            model: Trained model
            generator: Data generator
            num_samples: Number of samples to predict (None = all)
        
        Returns:
            Tuple of (predictions, true_labels)
        """
        if num_samples is None:
            # Use all samples
            steps = len(generator)
            total_samples = generator.n
        else:
            total_samples = num_samples
            steps = int(np.ceil(num_samples / generator.batch_size))
        
        all_predictions = []
        all_labels = []
        
        for i in range(steps):
            batch_images, batch_labels = next(generator)
            
            # Predictions (probabilities for all classes)
            predictions = model.predict(batch_images, verbose=0)
            all_predictions.extend(predictions)
            
            # Labels
            if len(batch_labels.shape) > 1:
                # One-hot encoded
                labels = np.argmax(batch_labels, axis=1)
            else:
                labels = batch_labels
            all_labels.extend(labels)
        
        return np.array(all_predictions[:total_samples]), np.array(all_labels[:total_samples])


if __name__ == "__main__":
    print("Data loader module loaded successfully")
