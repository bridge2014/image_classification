import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define parameters
img_height, img_width = 224, 224  # ResNet50 input size
batch_size = 32
epochs = 10  # Increase for better results
num_classes = 10  # e.g., 2 for benign/malignant; change to your number of cancer types
train_dir = '/data/erich/raj/data/train/'  # Path to training data
val_dir = '/data/erich/raj/data/test/'  # Path to validation data
test_dir = '/data/erich/raj/data/test/'  # Path to test data

# Step 2: Data generators with augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # 'binary' if num_classes=2
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Load pre-trained ResNet50 (without top classification layers)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers to use as feature extractor
for layer in base_model.layers:
    layer.trainable = False  # Later, unfreeze some for fine-tuning

# Step 4: Add custom layers for cancer classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Prevent overfitting
predictions = Dense(num_classes, activation='softmax')(x)  # Softmax for multi-class

model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # 'binary_crossentropy' for binary
              metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs
)

# Optional: Fine-tune by unfreezing top layers
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
    layer.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5, validation_data=val_generator)  # Fine-tune for a few epochs

# Step 7: Evaluate on test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# Step 8: Save the model
model.save('resnet50_cancer_classifier.h5')

# Step 9: Inference on a single pathology image
def predict_cancer_type(image_path, model_path='resnet50_cancer_classifier.h5'):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_labels = list(train_generator.class_indices.keys())  # Get labels from training generator
    return class_labels[class_idx], prediction[0][class_idx]

# Example usage
'''
image_path = 'path/to/your/pathology_image.jpg'
cancer_type, confidence = predict_cancer_type(image_path)
print(f"Predicted Cancer Type: {cancer_type} (Confidence: {confidence:.2f})")
'''

# Optional: Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()