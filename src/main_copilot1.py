import os
import math
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# =========================================
# Configuration
# =========================================
TRAIN_DIR = "/vast/home/fwang/image_ai/data/train/"   # <-- change to your path
TEST_DIR  = "/vast/home/fwang/image_ai/data/test/"    # <-- change to your path


IMG_SIZE = (224, 224)      # ResNet50 default
BATCH_SIZE = 32
SEED = 42

# Feature-extraction training
EPOCHS_STAGE1 = 10

# Fine-tuning training
EPOCHS_STAGE2 = 20
BASE_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

OUTPUT_DIR = "outputs1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_resnet50.keras")

# =========================================
# Optional: Mixed precision (if supported GPU)
# =========================================
# try:
#     from tensorflow.keras import mixed_precision
#     mixed_precision.set_global_policy("mixed_float16")
#     print("Using mixed precision.")
# except Exception as e:
#     print("Mixed precision not enabled:", e)

# =========================================
# Load Datasets
# =========================================
AUTOTUNE = tf.data.AUTOTUNE

# Create train/val split from TRAIN_DIR (keep a validation split for tuning)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",        # one-hot labels for 10 classes
    class_names=None,                # inferred from subfolders
    color_mode="rgb",                # converts grayscale to 3-channel automatically
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=train_ds.class_names,  # ensure same class ordering
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=train_ds.class_names,  # ensure same class ordering
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

class_names= train_ds.class_names
num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Cache & prefetch for performance
def configure(ds, training=False):
    ds = ds.cache()
    if training:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.prefetch(AUTOTUNE)

train_ds = configure(train_ds, training=True)
val_ds = configure(val_ds, training=False)
test_ds = configure(test_ds, training=False)

# =========================================
# Data Augmentation
# =========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),                # consider disabling for symmetric organs
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# =========================================
# Build the Model (ResNet50 backbone)
# =========================================
# Base model w/o top, pretrained on ImageNet
base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # Stage 1: freeze base

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = layers.Lambda(preprocess_input, name="resnet50_preprocess")(x)
x = base_model(x, training=False)  # Important: ensure not to update BN stats when frozen
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)         # regularization
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)  # cast back if mixed precision

model = models.Model(inputs, outputs, name="resnet50_medical_classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
    ],
)

model.summary()

# =========================================
# Callbacks
# =========================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(CHECKPOINT_PATH, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
]

# =========================================
# (Optional) Class Weights for Imbalance
# =========================================
# If classes are imbalanced, compute class weights from train set.
# Here is a quick utility-commented out. Uncomment and adapt if needed.
from collections import Counter
counts = Counter()
for _, y in train_ds.unbatch():
    cls_idx = int(tf.argmax(y).numpy())
    counts[cls_idx] += 1
total = sum(counts.values())
class_weight = {i: total/(num_classes*counts.get(i,1)) for i in range(num_classes)}
print("Class weights:", class_weight)
class_weight = class_weight

# class_weight = None

# =========================================
# Stage 1: Train classifier head (frozen base)
# =========================================
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# =========================================
# Stage 2: Fine-tune last blocks of ResNet50
# =========================================
# Unfreeze from a chosen layer (e.g., last ~50 layers)
fine_tune_at = 140  # ResNet50 has ~175 layers; adjust as needed
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# Re-compile with a lower LR
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
    ],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# =========================================
# Evaluation on TEST set
# =========================================
print("\nEvaluating on test set...")
test_metrics = model.evaluate(test_ds, verbose=1)
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name}: {val:.4f}")

# =========================================
# Per-class metrics & confusion matrix
# =========================================
# If sklearn isn't installed, pip install scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
import itertools

y_true = []
y_pred = []

for batch_images, batch_labels in test_ds:
    preds = model.predict(batch_images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion matrix:\n", cm)

# Plot confusion matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cm, class_names, normalize=True, title="Normalized Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix1.png"), bbox_inches="tight", dpi=150)
plt.close()

# =========================================
# Save final model
# =========================================
final_path = os.path.join(OUTPUT_DIR, "resnet50_medical_final1.keras")
model.save(final_path)
print("Saved final model to:", final_path)



# =========================================
# Inference on a new image (example)
# =========================================
def load_and_predict(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return {
        "pred_class": class_names[top_idx],
        "pred_index": top_idx,
        "probs": {class_names[i]: float(p) for i, p in enumerate(probs)}
    }

# Example usage:
# print(load_and_predict("path/to/single_image.png"))

# =========================================
# Grad-CAM for explainability (last conv layer)
# =========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the model output
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()
    return heatmap

def gradcam_on_image(img_path, last_conv_layer_name="conv5_block3_out", alpha=0.4):
    # Load image
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    arr = tf.keras.utils.img_to_array(img)
    input_arr = np.expand_dims(arr, axis=0)
    input_arr = preprocess_input(input_arr)

    heatmap = make_gradcam_heatmap(input_arr, model, last_conv_layer_name)
    heatmap = np.uint8(255 * heatmap)

    # Superimpose on original image
    import cv2
    img_bgr = cv2.cvtColor(np.uint8(arr), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    out_path = os.path.join(OUTPUT_DIR, "gradcam_overlay.png")
    cv2.imwrite(out_path, superimposed)
    return out_path

# Example:
# print("Grad-CAM saved to:", gradcam_on_image("path/to/single_image.png"))