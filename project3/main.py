import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 24
NUM_CLASSES = 10
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / "logs"

DATA_DIR = "/vast/projects/ebremer-group/fwang/image_classification/data"
TRAIN_DIR = DATA_DIR + "/train_split"
VAL_DIR = DATA_DIR + "/val_split"
TEST_DIR = DATA_DIR + "/test"


# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()  # keep kernels fast

def configure(ds, training, seed):
    ds = ds.cache()
    if training:
        ds = ds.shuffle(1000, seed=seed)
    return ds.prefetch(tf.data.AUTOTUNE)


# =========================================================
# Build datasets
# =========================================================
def build_datasets(data_dir, img_size, batch_size, val_split, seed):
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",               # one-hot for multi-class
        color_mode="rgb",
        image_size=(img_size, img_size),
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
        image_size=(img_size, img_size),
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
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
        class_names=train_ds.class_names,
    )

    return train_ds, val_ds, test_ds


def create_data_generators():
    """Create data generators with augmentation for training and validation."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def compute_class_weights(generator):
    """Compute class weights to handle imbalance."""
    class_indices = generator.class_indices
    class_labels = list(class_indices.keys())
    
    # Count samples per class
    class_counts = {}
    for class_name in class_labels:
        class_dir = Path(TRAIN_DIR) / class_name
        if class_dir.exists():
            class_counts[class_name] = len(list(class_dir.glob('*')))
        else:
            class_counts[class_name] = 0
    
    counts = [class_counts[label] for label in class_labels]
    classes = np.arange(NUM_CLASSES)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.repeat(classes, counts)
    )
    
    return dict(zip(classes, class_weights))

def build_model():
    """Build ResNet50 model with custom head."""
    # Load ResNet50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def fine_tune_model(model, base_model):
    """Unfreeze some layers for fine-tuning."""
    # Unfreeze the last 10 layers of ResNet50
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, class_names, save_path):
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    train_dir= '/vast/home/fwang/image_ai/data/train/'  # Path to training data folder
    test_dir = '/vast/home/fwang/image_ai/data/test/'    # Path to testing data folder
    
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print("TensorFlow:", tf.__version__)

    # Datasets
    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir, args.img_size, args.batch_size, args.val_split, args.seed
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Performance
    train_ds = configure(train_ds, training=True, seed=args.seed)
    val_ds   = configure(val_ds, training=False, seed=args.seed)
    test_ds  = configure(test_ds, training=False, seed=args.seed)
    
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    print("Computing class weights...")
    class_weights = compute_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    print("Building model...")
    model, base_model = build_model()
    
    # Initial training with frozen base
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODELS_DIR / 'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    print("Training initial model...")
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS // 2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Fine-tuning model...")
    model = fine_tune_model(model, base_model)
    
    # Continue training with fine-tuning
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=EPOCHS // 2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop]
    )
    
    # Save final model
    model.save(MODELS_DIR / 'final_model.h5')
    
    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    class_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Convert y_true to one-hot for ROC
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, RESULTS_DIR / 'confusion_matrix.png')
    
    # Plot ROC curves
    plot_roc_curves(y_true_onehot, y_pred_proba, class_names, RESULTS_DIR / 'roc_curves.png')
    
    print("Training complete. Results saved in results/ folder.")

if __name__ == "__main__":
    main()
