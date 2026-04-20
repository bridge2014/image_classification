import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 10
TRAIN_DIR = '/vast/home/fwang/image_ai/data/train/'
TEST_DIR = '/vast/home/fwang/image_ai/data/test/'

# ==========================================
# 2. DATA PREPARATION
# ==========================================
# Load datasets from directory structure
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False # Keep false for accurate Confusion Matrix later
)
# 2. CAPTURE CLASS NAMES NOW (while the attribute exists)
class_names = train_ds.class_names
label_map = {i: name for i, name in enumerate(class_names)}

# Optimize performance by buffering data from disk
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Save the Label Map (Index -> Class Name)
with open('medical_label_map.json', 'w') as f:
    json.dump(label_map, f, indent=4)

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
# Data Augmentation to prevent overfitting in medical scans
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
])

# Load ResNet50 Base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, CHANNELS))
base_model.trainable = False # Initial Freeze

# Final Model Assembly
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, CHANNELS)),
    data_augmentation,
    layers.Lambda(preprocess_input), # Normalize pixel values for ResNet
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Regularization
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# ==========================================
# 4. TRAINING: PHASE 1 (TRANSFER LEARNING)
# ==========================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting Phase 1: Training the Classification Head...")
history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# ==========================================
# 5. TRAINING: PHASE 2 (FINE-TUNING)
# ==========================================
# Unfreeze the last few layers of the ResNet base
base_model.trainable = True
for layer in base_model.layers[:140]: # Keep first 140 layers frozen
    layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

print("Starting Phase 2: Fine-Tuning the Base Model...")
history_fine = model.fit(
    train_ds, 
    validation_data=test_ds, 
    epochs=20, 
    initial_epoch=history.epoch[-1],
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)]
)

# ==========================================
# 6. EVALUATION & REPORTING
# ==========================================
# Generate predictions for the test set
y_true, y_pred = [], []
for imgs, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(model.predict(imgs, verbose=0), axis=1))

# Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Medical Classification Confusion Matrix')
plt.show()
print(" --- Save to file (before show()!) ----")
plt.savefig('Medical_Classification_Confusion_Matrix.png', dpi=150, bbox_inches='tight')


# Print Detailed Report
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# ==========================================
# 7. EXPORTING
# ==========================================
# Save for Python/Desktop
model.save('medical_resnet50_final.keras')

# Save model   
print(" --- Save model ----")
model.save('resnet50_cancer_classifier_gemini3.h5')

# Convert to TFLite for Mobile Deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantize
with open('medical_model_optimized.tflite', 'wb') as f:
    f.write(converter.convert())

print("Model saved in .keras and .tflite formats.")

# Step 8: Plot accuracy
print(" --- Step 9: Plot accuracy ----")
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize=(10, 8))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([10-1, 10-1], plt.ylim(), label='Start Fine Tuning') # Mark the transition
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True, alpha=0.3)
print(" --- Save to file (before show()!) ----")
plt.savefig('accuracy_plot_gemini3.png', dpi=150, bbox_inches='tight')
