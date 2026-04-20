# Setup
import os
import numpy as np
import itertools
import tensorflow as tf
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score


# Filter out corrupted images
'''
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")
'''

# Generate a Dataset
image_size = 224
batch_size = 32
seed=123
val_split = 0.2 
num_classes = 10
epochs = 10
input_shape=(image_size, image_size, 3)

# image dataset location
train_dir = "/vast/home/fwang/image_ai/data/train"
test_dir = "/vast/home/fwang/image_ai/data/test"
output_dir = "../output/"
os.makedirs(output_dir, exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",               # one-hot for multi-class
    color_mode="rgb",
    image_size=(image_size,image_size),
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="training",
)
    
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(image_size,image_size),
    batch_size=batch_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="validation",
    class_names=train_ds.class_names,       # ensure identical class order
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    image_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=False,
    class_names=train_ds.class_names,
)


class_names=train_ds.class_names
print (class_names)    


# Visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        #plt.title(int(labels[i]))
        #plt.title(int(labels[i].numpy()))
        plt.title(np.argmax(labels[i].numpy()))
        plt.axis("off")
        
# Using image data augmentation 
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

#############################
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images  
##########################
    
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        plt.axis("off")    
        
# Standardizing the data
'''   
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model   
'''

# Configure the dataset for performance   
# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)  


##########################################
def plot_confusion_matrix(cm, classes, out_path, normalize=True, title="Confusion Matrix"):
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9,
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()
##################################
def plot_multiclass_roc(y_true_oh, y_score, class_names, out_dir):
    """
    y_true_oh: (N, C) one-hot true labels (from test dataset)
    y_score:   (N, C) predicted probabilities (softmax)
    """
    os.makedirs(out_dir, exist_ok=True)
    num_classes = y_true_oh.shape[1]

    # Compute OvR ROC per class; skip classes absent in y_true
    fpr, tpr, roc_auc = {}, {}, {}
    present = []
    for i in range(num_classes):
        if y_true_oh[:, i].sum() == 0:
            continue
        present.append(i)
        fpr[i], tpr[i], _ = roc_curve(y_true_oh[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    if present:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_oh[:, present].ravel(), y_score[:, present].ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    else:
        fpr["micro"], tpr["micro"], roc_auc["micro"] = [np.array([0,1])]*2 + [np.nan]

    # macro-average
    if present:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in present]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in present:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(present)
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        fpr["macro"], tpr["macro"], roc_auc["macro"] = [np.array([0,1])]*2 + [np.nan]

    # sklearn convenience (macro/weighted AUC)
    try:
        macro_auc = roc_auc_score(y_true_oh, y_score, multi_class="ovr", average="macro")
        weighted_auc = roc_auc_score(y_true_oh, y_score, multi_class="ovr", average="weighted")
    except Exception:
        macro_auc, weighted_auc = roc_auc.get("macro", np.nan), np.nan

    # Combined plot
    plt.figure(figsize=(10, 8))
    for i in present:
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.55, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", linewidth=3,
             label=f"micro-average ROC (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], color="navy", linestyle=":", linewidth=3,
             label=f"macro-average ROC (AUC={roc_auc['macro']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (OvR)")
    plt.legend(loc="lower right", fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roc_multiclass.png")
    plt.savefig(out_path, dpi=160); plt.close()

    # Grid per-class
    cols = 5
    rows = int(np.ceil(max(1, len(present)) / cols))
    plt.figure(figsize=(3.2 * cols, 2.8 * rows))
    for idx, i in enumerate(present, start=1):
        ax = plt.subplot(rows, cols, idx)
        ax.plot(fpr[i], tpr[i], color="C0", lw=2, label=f"AUC={roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(class_names[i], fontsize=10)
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.grid(alpha=0.2)
        if idx % cols == 1: ax.set_ylabel("TPR")
        if idx > (rows - 1) * cols: ax.set_xlabel("FPR")
        ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "roc_per_class_grid.png")
    plt.savefig(out_path2, dpi=160); plt.close()

    print("\nAUC summary (OvR):")
    for i in present:
        print(f"  {class_names[i]}: AUC = {roc_auc[i]:.4f}")
    print(f"  micro-average AUC = {roc_auc['micro']:.4f}")
    print(f"  macro-average AUC = {roc_auc['macro']:.4f}")
    print(f"  sklearn macro AUC = {macro_auc:.4f} | sklearn weighted AUC = {weighted_auc:.4f}")
    print(f"ROC plots saved to: {out_path} and {out_path2}")
###################################################

# Build a model  
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)
#############################################

model = make_model(input_shape, num_classes)
keras.utils.plot_model(model, show_shapes=True)


# Train the model
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

# -------- Test evaluation --------
print("\nEvaluating on test set...")
test_metrics = model.evaluate(test_ds, verbose=1)
print(dict(zip(model.metrics_names, test_metrics)))

# Collect predictions & labels
y_true_oh = []
y_pred = []
y_score_chunks = []

for batch_images, batch_labels in test_ds:
    probs = model.predict(batch_images, verbose=0)
    y_score_chunks.append(probs)
    y_true_oh.append(batch_labels.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

y_score = np.vstack(y_score_chunks)         # (N, C)
y_true_oh = np.vstack(y_true_oh)            # (N, C)
y_true = np.argmax(y_true_oh, axis=1)       # (N,)

# Classification report
print("\nClassification Report (Test):")
print(classification_report(y_true, np.array(y_pred), target_names=class_names, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, np.array(y_pred), labels=np.arange(num_classes))
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plot_confusion_matrix(cm, class_names, out_path=cm_path,
                      normalize=True, title="Normalized Confusion Matrix (Test)")
print(f"Saved confusion matrix to {cm_path}")

# ROC / AUC plots
plot_multiclass_roc(y_true_oh, y_score, class_names, output_dir)

# Save final model
final_path = os.path.join(output_dir, "final.keras")
model.save(final_path)
print(f"Saved final model to {final_path}")
 
 
print(" --- Save Accuracy Epoch chart ----")
'''
acc = history.history['accuracy']
val_acc = history.history['val_accuracy'] 
plt.figure(figsize=(8, 8))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning') # Mark the transition
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')    
plt.grid(True, alpha=0.3)   
out_path = os.path.join(output_dir, "accuracy_plot_roc_tf1.png") 
plt.savefig(out_path, dpi=150, bbox_inches='tight')  
'''

# Step 8: Plot accuracy
print(" --- Plot accuracy ----")
plt.figure(figsize=(10, 5))           # optional: make it nicer size
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')  
plt.grid(True, alpha=0.3)
out_path = os.path.join(output_dir, "accuracy_plot.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight') 
    
# Run inference on new data
'''
img = keras.utils.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
'''