import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = (224, 224) # ResNet50 default input size
BATCH_SIZE = 32

# Path to training data folder
train_dir = '/vast/home/fwang/image_ai/data/train/'  
# Path to testing data folder
test_dir = '/vast/home/fwang/image_ai/data/test/'

# Load training and test sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 1. Load base model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Freeze the base model (don't train existing ResNet weights yet)
base_model.trainable = False

#### Updated Model with Augmentation #####
# 1. Define the Augmentation Pipeline
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"), # Useful if orientation doesn't matter
    layers.RandomRotation(0.2), # Rotates by +/- 20%
    layers.RandomZoom(0.2),     # Zooms in/out by 20%
    layers.RandomContrast(0.1), # Helpful for varying exposure in medical scans
])

# 2. Re-build the Model with Augmentation
model = models.Sequential([
    # Input layer
    layers.InputLayer(input_shape=(224, 224, 3)),
    
    # Add Augmentation first
    data_augmentation,
    
    # Preprocessing (ResNet specific scaling)
    layers.Lambda(preprocess_input),
    
    # The Pre-trained Base
    base_model,
    
    # The Classification Head
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), 
    layers.Dense(10, activation='softmax') # 10 classes
])

# Re-compile and train as before
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

### Adding a Learning Rate Scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,   # Multiply LR by 0.2
    patience=3,    # Wait 3 epochs before dropping
    min_lr=1e-6    # Don't go below this
)

# Train the custom head
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=initial_epochs,
    callbacks=[reduce_lr]
)

# Evaluate performance
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# Assuming you have already run the initial training from the previous step...

initial_epochs=10

# 1. Unfreeze the base model
base_model.trainable = True

# 2. Refreeze everything EXCEPT the last few layers
# ResNet50 has 175 layers. We'll unfreeze from layer 140 onwards.
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# 3. Recompile with a VERY low learning rate
# A low learning rate (1e-5) prevents the model from "forgetting" what it learned
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Continue training (Fine-tuning)
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs # Adding to previous training

history_fine = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=total_epochs,    
    initial_epoch=history.epoch[-1], # Start from where the last training ended
    callbacks=[reduce_lr]
)

# Save model   
print(" --- Save model ----")
model.save('resnet50_cancer_classifier_gemini.h5')

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([10-1, 10-1], plt.ylim(), label='Start Fine Tuning') # Mark the transition
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.grid(True, alpha=0.3)
# Save to file (before show()!)
print(" --- Save to file (before show()!) ----")
plt.savefig('accuracy_plot_gemini.png', dpi=150, bbox_inches='tight')