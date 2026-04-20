"""
Medical Image Classification using ResNet50
Trains a ResNet50 model on medical images with 10 classes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MedicalImageClassifier:
    """Medical image classifier using ResNet50."""
    
    def __init__(
        self,
        train_dir: str,
        test_dir: str,
        num_classes: int = 10,
        img_height: int = 224,
        img_width: int = 224,
        batch_size: int = 32
    ):
        """
        Initialize the classifier.
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory
            num_classes: Number of classification classes
            img_height: Image height (ResNet50 default: 224)
            img_width: Image width (ResNet50 default: 224)
            batch_size: Batch size for training
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def create_data_generators(self):
        """Create data generators with augmentation for training."""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            validation_split=0.2  # 20% for validation
        )
        
        # Test data - only rescaling
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        self.val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Store class names
        self.class_names = list(self.train_generator.class_indices.keys())
        print(f"\nFound {len(self.class_names)} classes: {self.class_names}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
    def build_model(self, fine_tune: bool = False, fine_tune_at: int = 100):
        """
        Build ResNet50 model for medical image classification.
        
        Args:
            fine_tune: Whether to fine-tune the base model
            fine_tune_at: Layer index from which to start fine-tuning
        """
        # Load pre-trained ResNet50 (without top layers)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Preprocessing for ResNet50
        x = keras.applications.resnet50.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs)
        
        print("\nModel architecture created")
        print(f"Total layers in base model: {len(base_model.layers)}")
        
        # Optionally fine-tune the base model
        if fine_tune:
            base_model.trainable = True
            # Fine-tune from fine_tune_at layer onwards
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print(f"Fine-tuning enabled from layer {fine_tune_at}")
        
    def compile_model(self, learning_rate: float = 0.0001):
        """Compile the model with optimizer and loss function."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        print("\nModel compiled")
        
    def get_callbacks(self, model_save_path: str = 'best_model.h5'):
        """Create training callbacks."""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        return callbacks
        
    def train(self, epochs: int = 50, model_save_path: str = 'best_model.h5'):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            model_save_path: Path to save the best model
        """
        print(f"\nStarting training for {epochs} epochs...")
        
        callbacks = self.get_callbacks(model_save_path)
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
    def evaluate(self):
        """Evaluate model on test set."""
        print("\nEvaluating on test set...")
        
        test_loss, test_accuracy, test_auc = self.model.evaluate(
            self.test_generator,
            verbose=1
        )
        
        print(f"\nTest Results:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"AUC: {test_auc:.4f}")
        
        return test_loss, test_accuracy, test_auc
        
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training and validation metrics."""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
        
    def predict(self, image_path: str):
        """
        Predict class for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Predicted class name and probabilities
        """
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_height, self.img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
        
    def save_model(self, path: str = 'final_model.h5'):
        """Save the model."""
        self.model.save(path)
        print(f"\nModel saved to {path}")
        
    def load_model(self, path: str):
        """Load a saved model."""
        self.model = keras.models.load_model(path)
        print(f"\nModel loaded from {path}")
        
    def get_confusion_matrix(self):
        """Generate confusion matrix on test set."""
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        # Get predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_claude.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved to confusion_matrix.png")
        plt.close()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            true_classes,
            predicted_classes,
            target_names=self.class_names
        ))


def main():
    """Main execution function."""
    
    # Configure paths
    TRAIN_DIR = '/vast/home/fwang/image_ai/data/train/'  
    TEST_DIR = '/vast/home/fwang/image_ai/data/test/' 
       
    num_classes=10
    batch_size=32
    fine_tune=False
    learning_rate=0.0001
    epochs=50
    model_save_path='best_resnet50_medical_claude.h5'
    final_model_name='final_resnet50_medical_claude.h5'
    
    # Check if directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found: {TRAIN_DIR}")
        print("Please update TRAIN_DIR with your training data path")
        return
    
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found: {TEST_DIR}")
        print("Please update TEST_DIR with your test data path")
        return
    
    # Create classifier
    classifier = MedicalImageClassifier(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    # Prepare data
    classifier.create_data_generators()
    
    # Build model
    classifier.build_model(fine_tune=fine_tune)
    
    # Display model summary
    classifier.model.summary()
    
    # Compile model
    classifier.compile_model(learning_rate=learning_rate)
    
    # Train model
    classifier.train(epochs=epochs, model_save_path=model_save_path)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate on test set
    classifier.evaluate()
    
    # Generate confusion matrix
    classifier.get_confusion_matrix()
    
    # Save final model
    classifier.save_model(final_model_name)
    
    # Example: Predict on a single image
    # test_image_path = 'path/to/test/image.jpg'
    # predicted_class, confidence, all_probs = classifier.predict(test_image_path)
    # print(f"\nPredicted: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()