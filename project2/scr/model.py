"""
Model architecture and utilities for medical image classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import config


class MedicalImageClassifier:
    """
    ResNet50-based medical image classifier with fine-tuning capabilities
    """

    def __init__(self, num_classes=config.NUM_CLASSES, input_shape=config.IMAGE_SIZE):
        self.num_classes = num_classes
        self.input_shape = (*input_shape, 3)  # Add 3 channels for RGB
        self.model = None
        self.initial_model = None

    def build_model(self):
        """
        Build ResNet50 model with custom top layers
        
        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        print("Building ResNet50 model...")

        # Load pre-trained ResNet50
        base_model = ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'  # Use ImageNet pre-trained weights
        )

        # Freeze base model initially
        base_model.trainable = False

        # Build the model
        inputs = layers.Input(shape=self.input_shape)

        # Data augmentation layer (additional augmentation at input)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)

        # Normalization layer
        x = layers.Rescaling(1./127.5, offset=-1)(x)

        # Pass through base model (ResNet50)
        x = base_model(x, training=False)

        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Dropout for regularization
        x = layers.Dropout(0.5)(x)

        # Dense layers for classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.initial_model = self.model

        print(f"Model created with {len(self.model.layers)} layers")
        print(f"Total parameters: {self.model.count_params():,}")

        return self.model

    def compile_model(self, learning_rate=config.LEARNING_RATE):
        """
        Compile the model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
        )

        print(f"Model compiled with learning rate: {learning_rate}")

    def get_model(self):
        """Get the current model"""
        return self.model

    def summary(self):
        """Print model summary"""
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        self.model.summary()

    def unfreeze_base_model(self, num_layers_to_unfreeze=50):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            num_layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        print(f"\nUnfreezing last {num_layers_to_unfreeze} layers for fine-tuning...")

        # Get base model
        base_model = self.model.layers[4]  # ResNet50 is at index 4
        base_model.trainable = True

        # Freeze all but last num_layers_to_unfreeze layers
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

        # Count trainable parameters
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        print(f"Trainable parameters: {trainable_params:,}")

    def get_callbacks(self):
        """
        Get training callbacks
        
        Returns:
            list: List of callback objects
        """
        callbacks = [
            # Save best model based on validation loss
            tf.keras.callbacks.ModelCheckpoint(
                config.BEST_MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate when validation loss plateaus
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),

            # Log to tensorboard
            tf.keras.callbacks.TensorBoard(
                log_dir=f'{config.RESULTS_DIR}/logs',
                histogram_freq=1,
                write_graph=True
            )
        ]

        return callbacks
