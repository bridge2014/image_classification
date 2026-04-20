import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import warnings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0 = all logs, 1 = filter info, 2 = filter info+warning, 3 = filter all

# Just ignore / silence the warning
warnings.filterwarnings(
    "ignore",
    message="Your `PyDataset` class should call",
    category=UserWarning
)


# Simple example: Train a linear model on random data
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))  # Should show GPUs if requested


# Optional: Check and use GPU (TensorFlow auto-detects; this confirms)
print(" --- Optional: Check and use GPU (TensorFlow auto-detects; this confirms). ----")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found; using CPU.")

# Step 1: Define parameters
print(" --- Step 1: Define parameters. ----")
img_height, img_width = 224, 224  # ResNet50 input size
batch_size = 32
epochs = 60  # Increase as needed
num_classes = 10  # Your 10 cancer types
train_dir = '/vast/home/fwang/image_ai/data/train/'  # Path to training data folder
test_dir = '/vast/home/fwang/image_ai/data/test/'    # Path to testing data folder


# Step 2: Prepare data generators (split train into train/val)
# First, get all training image paths and labels
print(" --- Step 2: Prepare data generators (split train into train/val) First, get all training image paths and labels ----")
train_images = []
train_labels = []
class_names = sorted(os.listdir(train_dir))  # e.g., ['class_0', ..., 'class_9']
class_indices = {name: idx for idx, name in enumerate(class_names)}

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    for img_name in os.listdir(class_path):
        train_images.append(os.path.join(class_path, img_name))
        train_labels.append(class_indices[class_name])

# Split into train and validation (80/20)
train_imgs, val_imgs, train_labs, val_labs = train_test_split(
    train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# ------------------------------------------------
# Option A: Apply augmentation only to training data (recommended)
# ------------------------------------------------

# Use ImageDataGenerator for augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input ,
    validation_split   = 0.10 # ? 10% ? validation
)

# Validation: only rescaling (very important!)
valid_datagen = ImageDataGenerator(
    rescale        = 1./255,
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input ,
    validation_split = 0.10
)

test_datagen = ImageDataGenerator(rescale=1./255)

# For split data, use flow_from_dataframe or custom generator; here, we'll use flow_from_directory after temp folders (simpler alternative: use flow)
# But for simplicity with split, create temp val folders (or use flow with subset, but requires same structure)
# Alternative: Use flow_from_directory with validation_split=0.2 directly on train_dir
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',  # 90% for training
    seed=42
)
print(f"Training samples   : {train_generator.samples}")

val_generator = valid_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # 10% for validation
    seed=42
)
print(f"Validation samples : {val_generator.samples}")

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # For accurate predictions
)
print(f"Test samples : {test_generator.samples}")

# Step 3: Load pre-trained ResNet50
print(" --- Step 3: Load pre-trained ResNet50 ----")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers
print(" --- Freeze base layers ----")
for layer in base_model.layers:
    layer.trainable = False

# Step 4: Add custom layers for 10 classes
print(" --- Step 4: Add custom layers for 10 classes ----")
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile and train
print(" --- Step 5: Compile and train ----")
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,    
    validation_data=val_generator,    
    epochs=epochs
)

length_base_model_layers = len(base_model.layers);
print(f" === length_base_model_layers === : {length_base_model_layers}")

# Optional: Fine-tune (unfreeze last 20 layers for better adaptation)
print(" --- Optional: Fine-tune (unfreeze last 20 layers for better adaptation)  ----")
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=20, validation_data=val_generator)  # Additional epochs

# Step 6: Evaluate on test set
print(" --- Step 6: Evaluate on test set  ----")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)

# Step 7: Save model   
print(" --- Step 7: Save model ----")
model.save('resnet50_cancer_classifier5.h5')

# Step 8: Plot accuracy
print(" --- Step 9: Plot accuracy ----")
plt.figure(figsize=(10, 5))           # optional: make it nicer size
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
# Save to file (before show()!)
print(" --- Save to file (before show()!) ----")
plt.savefig('accuracy_plot5.png', dpi=150, bbox_inches='tight')
#plt.show()