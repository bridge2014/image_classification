import os
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Parameters
IMG_SIZE = (224, 224) # ResNet50 default input size
BATCH_SIZE = 32
NUM_CLASSES = 10

# Path to training data folder
train_dir = '/vast/home/fwang/image_ai/data/train/'  
# Path to testing data folder
test_dir = '/vast/home/fwang/image_ai/data/test/'


# ==========================================
# 2. DATA PREPARATION
# ==========================================
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

# 2. CAPTURE CLASS NAMES NOW (while the attribute exists)
class_names = train_ds.class_names
label_map = {i: name for i, name in enumerate(class_names)}

# Optimize performance by buffering data from disk
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Save the Label Map (Index -> Class Name)
with open('medical_label_map_gemini_new.json', 'w') as f:
    json.dump(label_map, f, indent=4)


# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
# 1. Load base model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 2. Freeze the base model (don't train existing ResNet weights yet)
base_model.trainable = False

# 3. Build the final model
model = models.Sequential([
    layers.Lambda(preprocess_input, input_shape=(224, 224, 3)), # Built-in ResNet preprocessing
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting in medical data
    layers.Dense(NUM_CLASSES, activation='softmax') # 10 classes
])

# ==========================================
# 4. TRAINING: PHASE 1 (TRANSFER LEARNING)
# ==========================================
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

initial_epochs=10

# Train the custom head
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=initial_epochs
)

# Evaluate performance
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy*100:.2f}%")


# ==========================================
# 5. TRAINING: PHASE 2 (FINE-TUNING)
# ==========================================
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
    initial_epoch=history.epoch[-1] # Start from where the last training ended
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
#plt.show()
print(" --- Save to file (before show()!) ----")
plt.savefig('Medical_Classification_Confusion_Matrix_gemini_new.png', dpi=150, bbox_inches='tight')

# Print Detailed Report
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


# ==========================================
# 7. EXPORTING
# ==========================================
# Save model   
print(" --- Save model ----")
model.save('resnet50_cancer_classifier_gemini_new.h5')

print(" --- Save Accuracy Epoch chart ----")
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
plt.figure(figsize=(8, 8))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([10-1, 10-1], plt.ylim(), label='Start Fine Tuning') # Mark the transition
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
#plt.show()
plt.grid(True, alpha=0.3)
print(" --- Save to file (before show()!) ----")
plt.savefig('accuracy_plot_gemini_new.png', dpi=150, bbox_inches='tight')