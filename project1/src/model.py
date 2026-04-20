"""
Model Architecture Module
Building and configuring ResNet50 model with transfer learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class ResNet50Classifier:
    """
    ResNet50-based medical image classifier with transfer learning
    """
    
    def __init__(self, num_classes, input_shape=(224, 224, 3), 
                 l2_reg=1e-4, dropout_rate=0.5):
        """
        Initialize the classifier
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape
            l2_reg: L2 regularization parameter
            dropout_rate: Dropout rate
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.model = None
        self.base_model = None
    
    # In model.py, add more sophisticated layers
    def build_enhanced_model(self):
        # ... existing base model code ...
        
        # Add attention mechanism or SE blocks
        x = layers.GlobalAveragePooling2D()(x)
        
        # Feature enrichment
        x = layers.Dense(1024, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.6)(x)
        
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        return keras.Model(inputs=inputs, outputs=outputs)
    
        
    def build_model(self):
        """
        Build the model with transfer learning
        
        Returns:
            Compiled Keras model
        """
        print("\n[MODEL] Building ResNet50 model...")
        
        # Load pretrained ResNet50
        self.base_model = ResNet50(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        # Freeze base model weights initially
        self.base_model.trainable = False
        
        # Build custom top layers
        inputs = keras.Input(shape=self.input_shape)
        
        # Base model
        x = self.base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with regularization
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=l2(self.l2_reg),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(self.l2_reg),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate * 0.6)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        print(f"[OK] Model built successfully!")
        print(f"[INFO] Total parameters: {self.model.count_params():,}")
        
        #return self.model
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
                
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        print(f"\n[CONFIG] Compiling model with learning rate {learning_rate}...")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("[OK] Model compiled successfully!")
    
    def unfreeze_base_model(self, from_layer=100):
        """
        Unfreeze base model layers for fine-tuning
        
        Args:
            from_layer: Layer number to start unfreezing from
        """
        
        if self.base_model is None:
            raise ValueError("Base model not initialized.")
        
        print(f"\n[FINETUNE] Unfreezing base model from layer {from_layer}...")
        
        # Find the base model in the compiled model
        base_model_layer = None
        
        if self.model is not None:
          for layer in self.model.layers:
              if layer.name == 'resnet50':
                base_model_layer = layer
                break
        
        if base_model_layer is None:
            # If not found by name, try to access it directly
            base_model_layer = self.base_model
        
        # Freeze early layers
        for layer in base_model_layer.layers[:from_layer]:
            layer.trainable = False
        
        # Unfreeze later layers
        for layer in base_model_layer.layers[from_layer:]:
            layer.trainable = True
        
        print(f"[OK] Unfroze {len(base_model_layer.layers) - from_layer} layers for fine-tuning")
        print(f"[INFO] Total trainable parameters: {self.model.count_params():,}")
    
    def get_callbacks(self, model_checkpoint_path, patience=5, 
                     factor=0.5, min_lr=1e-7):
        """
        Get training callbacks
        
        Args:
            model_checkpoint_path: Path to save best model
            patience: Early stopping patience
            factor: LR reduction factor
            min_lr: Minimum learning rate
        
        Returns:
            List of callbacks
        """
        
        # Replace ReduceLROnPlateau with cosine annealing
        def cosine_annealing(epoch, lr):
            initial_lr = 1e-4
            epochs = 50
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        
        #lr_callback = keras.callbacks.LearningRateScheduler(cosine_annealing)


        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=factor,
                patience=patience - 2,
                min_lr=min_lr,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode='max'
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def get_model_summary(self):
        """Print model summary"""
        
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        self.model.summary()
        print("="*70 + "\n")
    
    def save_model(self, filepath):
        """
        Save model
        
        Args:
            filepath: Path to save model
        """
        
        if self.model is None:
            raise ValueError("Model not built yet.")
        
        self.model.save(filepath)
        print(f"[OK] Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load model
        
        Args:
            filepath: Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        print(f"[OK] Model loaded from: {filepath}")
        return self.model ;


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

'''
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1 + 1e-8)) \
               -tf.reduce_sum((1-alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0 + 1e-8))
    return focal_loss_fixed
'''
'''
# Use Keras Tuner for automated hyperparameter optimization
import keras_tuner as kt

def build_model_with_tuning(hp):
    # Tune learning rate, dropout rates, layer sizes, etc.
    lr = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
    dropout = hp.Float('dropout', 0.2, 0.6, step=0.1)
    # ... build model with tuned parameters ...

tuner = kt.Hyperband(build_model_with_tuning, ...)
'''

# Use in compile
#model.compile(loss=focal_loss(), ...)

# Add more regularization techniques
#from tensorflow.keras.regularizers import l1_l2

# Use L1 + L2 regularization
#kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)

# Add label smoothing
#loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance
    
    Args:
        gamma: Focusing parameter
        alpha: Weighting factor
    
    Returns:
        Focal loss function
    """
    def focal_crossentropy(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        alpha_t = alpha
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_loss_value = alpha_t * focal_weight * ce
        
        return tf.reduce_mean(focal_loss_value)
    
    return focal_crossentropy


if __name__ == "__main__":
    # Test model building
    classifier = ResNet50Classifier(num_classes=10)
    model = classifier.build_model()
    classifier.compile_model()
    classifier.get_model_summary()
