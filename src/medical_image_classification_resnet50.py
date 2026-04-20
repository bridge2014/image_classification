import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(train_dir, test_dir, img_size=224, batch_size=32):
    """
    Load and preprocess medical images using ImageDataGenerator
    
    Args:
        train_dir: Path to training dataset directory
        test_dir: Path to test dataset directory
        img_size: Image size for ResNet50 (default 224x224)
        batch_size: Batch size for training
    
    Returns:
        train_generator, test_generator, num_classes
    """
    
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # Only rescaling for test set (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class labels: {train_generator.class_indices}")
    
    return train_generator, test_generator, num_classes


# ============================================================================
# 2. BUILD RESNET50 MODEL
# ============================================================================

def build_resnet50_model(num_classes, learning_rate=0.001):
    """
    Build ResNet50 model with transfer learning
    
    Args:
        num_classes: Number of output classes (10 in your case)
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    
    # Load pre-trained ResNet50 (ImageNet weights)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers (transfer learning)
    base_model.trainable = False
    
    # Build custom top layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


# ============================================================================
# 3. FINE-TUNING FUNCTION
# ============================================================================

def fine_tune_model(model, base_model, learning_rate=0.0001):
    """
    Unfreeze base model layers and fine-tune with lower learning rate
    
    Args:
        model: The compiled model
        base_model: The base ResNet50 model
        learning_rate: Lower learning rate for fine-tuning
    
    Returns:
        Model ready for fine-tuning
    """
    
    # Unfreeze base model layers
    base_model.trainable = True
    
    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# 4. TRAINING FUNCTION
# ============================================================================

def train_model(model, train_generator, test_generator, epochs=50, 
                early_stopping=True, checkpoint=True):
    """
    Train the model with callbacks
    
    Args:
        model: Compiled model
        train_generator: Training data generator
        test_generator: Validation/test data generator
        epochs: Number of epochs
        early_stopping: Whether to use early stopping
        checkpoint: Whether to save best model
    
    Returns:
        Training history
    """
    
    callbacks = []
    
    # Early stopping callback
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ))
    
    # Model checkpoint callback
    if checkpoint:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            'best_resnet50_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ))
    
    # Learning rate reduction callback
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ))
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# ============================================================================
# 5. EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, test_generator):
    """
    Evaluate model on test dataset
    
    Args:
        model: Trained model
        test_generator: Test data generator
    
    Returns:
        Test loss and accuracy
    """
    
    print("\nEvaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return test_loss, test_accuracy


# ============================================================================
# 6. PREDICTION FUNCTION
# ============================================================================

def predict_single_image(model, image_path, class_labels, img_size=224):
    """
    Make prediction on a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_labels: Dictionary of class labels
        img_size: Image size
    
    Returns:
        Predicted class and confidence
    """
    
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Reverse class indices to get class name
    class_idx_to_label = {v: k for k, v in class_labels.items()}
    predicted_class_name = class_idx_to_label[predicted_class_idx]
    
    return predicted_class_name, confidence


# ============================================================================
# 7. VISUALIZATION FUNCTION
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history object
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set paths to your datasets
    TRAIN_DIR = "path/to/train_dataset"
    TEST_DIR = "path/to/test_dataset"
    
    print("=" * 70)
    print("Medical Image Classification using ResNet50")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    train_gen, test_gen, num_classes = load_and_preprocess_data(
        TRAIN_DIR, 
        TEST_DIR,
        img_size=224,
        batch_size=32
    )
    
    # Step 2: Build model
    print("\n[Step 2] Building ResNet50 model...")
    model, base_model = build_resnet50_model(num_classes=num_classes, learning_rate=0.001)
    print(model.summary())
    
    # Step 3: Train initial model with frozen base
    print("\n[Step 3] Training model (frozen base layers)...")
    history = train_model(
        model,
        train_gen,
        test_gen,
        epochs=20,
        early_stopping=True,
        checkpoint=True
    )
    
    # Step 4: Fine-tune model
    print("\n[Step 4] Fine-tuning model (unfreezing base layers)...")
    model = fine_tune_model(model, base_model, learning_rate=0.0001)
    
    history_finetune = train_model(
        model,
        train_gen,
        test_gen,
        epochs=20,
        early_stopping=True,
        checkpoint=True
    )
    
    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model...")
    test_loss, test_accuracy = evaluate_model(model, test_gen)
    
    # Step 6: Plot training history
    print("\n[Step 6] Plotting training history...")
    plot_training_history(history_finetune)
    
    # Step 7: Save final model
    print("\n[Step 7] Saving model...")
    model.save('final_resnet50_model.h5')
    
    # Step 8: Make predictions on single images (optional)
    print("\n[Step 8] Making predictions on sample images...")
    sample_image_path = "path/to/sample/image.jpg"
    class_labels = train_gen.class_indices
    
    predicted_class, confidence = predict_single_image(
        model, 
        sample_image_path, 
        class_labels
    )
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print("=" * 70)