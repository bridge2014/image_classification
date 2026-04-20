import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import warnings


# Parameters
IMG_SIZE = (224, 224) # ResNet50 default input size
BATCH_SIZE = 32

# Path to training data folder
train_dir = '/vast/home/fwang/image_ai/data/train/'  
# Path to testing data folder
test_dir = '/vast/home/fwang/image_ai/data/test/'

class_names = sorted(os.listdir(train_dir))  # e.g., ['class_0', ..., 'class_9']

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

# 3. Build the final model
model = models.Sequential([
    layers.Lambda(preprocess_input, input_shape=(224, 224, 3)), # Built-in ResNet preprocessing
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting in medical data
    layers.Dense(10, activation='softmax') # 10 classes
])

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


# Assuming you have already run the initial training from the previous step...

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

# Step 6: Evaluate on test set
print(" --- Step 6: Evaluate on test set  ----")
predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_ds.classes
print(classification_report(true_classes, predicted_classes, target_names=class_names))


# Step 6: Evaluate on test set
print(" --- Step 6: Evaluate on test set  ----")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
print(classification_report(true_classes, predicted_classes, target_names=class_names))



# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)

# Save model   
print(" --- Save model ----")
model.save('resnet50_cancer_classifier_gemini1.h5')


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
#plt.show()
plt.grid(True, alpha=0.3)
print(" --- Save to file (before show()!) ----")
plt.savefig('accuracy_plot_gemini1.png', dpi=150, bbox_inches='tight')