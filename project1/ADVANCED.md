# Medical Image Classification - Advanced Documentation

## Architecture Deep Dive

### ResNet50 Transfer Learning

ResNet50 (Residual Network with 50 layers) is a pre-trained deep neural network that has learned to extract powerful image features from ImageNet. This project uses transfer learning to adapt these features for medical image classification.

#### Why ResNet50?

1. **Proven Architecture**: Deep residual networks solve the vanishing gradient problem
2. **Pre-trained Weights**: 25M parameters already learned from 1.2M ImageNet images
3. **Good Performance/Speed Trade-off**: Faster than larger models like ResNet152
4. **Fine-tuning Friendly**: Intermediate layers can be easily unfrozen for domain adaptation

#### Architecture Layers

```
Input (224, 224, 3)
  ↓
Convolution (7×7, stride=2)
  ↓
ResNet50 Backbone (frozen/unfrozen)
  ├── Layer 1: 3 residual blocks (~64 channels)
  ├── Layer 2: 4 residual blocks (~128 channels)
  ├── Layer 3: 6 residual blocks (~256 channels)
  └── Layer 4: 3 residual blocks (~512 channels)
  ↓
GlobalAveragePooling2D (7×7×512 → 512)
  ↓
Dense(512, ReLU)
  ↓
BatchNormalization
  ↓
Dropout(0.5)
  ↓
Dense(256, ReLU)
  ↓
BatchNormalization
  ↓
Dropout(0.3)
  ↓
Dense(10, Softmax) → Output probabilities
```

### Custom Top Layers

The custom layers added on top serve specific purposes:

- **Dense(512)**: Learns interpretable combinations of ResNet50 features
- **BatchNormalization**: Stabilizes training, reduces internal covariate shift
- **Dropout(0.5)**: Prevents overfitting in the dense layer
- **Dense(256)**: Further feature combination with regularization
- **Dropout(0.3)**: Lighter dropout for the final hidden layer
- **Dense(10, Softmax)**: Produces probability distribution over 10 classes

## Training Strategy

### Phase 1: Frozen Base Model Training (20 epochs)

**Objective**: Learn class-specific patterns while keeping medical feature representations fixed

**Configuration**:
- ResNet50 base: All weights frozen
- Custom layers: All weights trainable
- Learning rate: 1e-4 (moderate pace)
- Optimizer: Adam (adaptive learning rate)
- Batch size: 32 images

**Why freeze the base?**
- Avoids overfitting with limited medical data
- Preserves general feature extraction ability
- Faster training (fewer parameters to update)
- Good starting point for fine-tuning

### Phase 2: Fine-tuning (10 epochs)

**Objective**: Adapt ResNet50 layers to medical imaging domain

**Configuration**:
- ResNet50 base: Unfreeze from layer 100 onwards
  - Freezes: Early layers (edges, textures, basic shapes)
  - Unfreezes: Late layers (complex patterns, domain-specific features)
- Learning rate: 1e-5 (10x lower, conservative updates)
- Why lower LR? Prevents catastrophic forgetting of pre-trained knowledge

**Which layers to unfreeze?**

ResNet50 layers:
- Layer 1 (frozen): Generic edge + texture detectors
- Layer 2 (frozen): Basic shape patterns
- Layer 3 (frozen): Object parts
- Layer 4 (unfrozen): Domain-specific complex patterns → Medical imaging adaptations

Unfreezing from layer 100 means approximately 200+ million learned weights in early layers are frozen, and only ~4.7M parameters in later layers are fine-tuned.

## Data Augmentation Strategy

### Why Augmentation Matters for Medical Imaging?

Medical imaging datasets are typically smaller than natural image datasets:
- Limited data availability (privacy, cost)
- Class imbalance (rare diseases)
- Similar appearance between classes

### Augmentation Techniques Applied

```python
AUGMENTATION_CONFIG = {
    'rotation': 20,              # ±20° rotations
    'width_shift': 0.2,         # ±20% horizontal shift
    'height_shift': 0.2,        # ±20% vertical shift
    'shear': 0.2,               # ±20% shear transformation
    'zoom': 0.2,                # ±20% zoom
    'horizontal_flip': True,     # Mirror horizontally
    'vertical_flip': False,      # No vertical flip (medical images have orientation)
}
```

### Augmentation Effects on Medical Images

1. **Rotation**: Accounts for different scanning angles
2. **Shifting**: Simulates different patient positioning
3. **Shearing**: Varies image perspective
4. **Zooming**: Tests scale invariance
5. **Horizontal Flip**: Accounts for bilateral symmetry in medical imaging
6. **No Vertical Flip**: Preserves anatomical orientation (important for medical images)

### Applied Only to Training Data

- **Training**: Augmented data (constant variation)
- **Validation**: Original data only (consistent evaluation)
- **Test**: Original data only (final evaluation)

## Class Imbalance Handling

### The Problem

Medical imaging datasets often have class imbalance:
```
Class 1 (Common disease): 2000 images
Class 2 (Common disease): 1800 images
...
Class 9 (Rare disease):    50 images ← 40x less data!
Class 10 (Rare disease):   30 images ← 67x less data!
```

### Solution: Class Weights

**Automatic calculation** based on inverse frequency:

```
weight[i] = total_samples / (num_classes * samples[i])
```

**Effect**:
- Rare classes get higher weights (more penalty if wrong)
- Common classes get lower weights (less penalty)
- Model pays attention to minority classes

**Example weights**:
```
Class 1: 8000 / (10 × 2000) = 0.4   (down-weighted)
Class 9: 8000 / (10 × 50) = 16.0    (heavily up-weighted)
```

### Application

During training:
```python
model.fit(
    train_generator,
    class_weight=class_weights,  # Applied here
    ...
)
```

Each sample from rare class contributes 40x more to loss calculation.

## Regularization Techniques

### 1. L2 Regularization (Weight Decay)

- **Purpose**: Prevents weights from growing too large
- **Effect**: Shrinks less important weights toward zero
- **Implementation**: `kernel_regularizer=L2(1e-4)`
- **Impact**: Smoother decision boundaries, better generalization

### 2. Dropout

**After Dense(512)**:
- **Dropout rate**: 0.5 (drop 50% of neurons)
- **Effect**: Prevents co-adaptation of neurons
- **Result**: Acts like ensemble of smaller networks

**After Dense(256)**:
- **Dropout rate**: 0.3 (drop 30% of neurons)
- **Effect**: Less aggressive dropout for final layer
- **Result**: Balances regularization with model capacity

### 3. Batch Normalization

- **Purpose**: Normalize layer inputs
- **Effect**: Reduces internal covariate shift
- **Benefits**:
  - Faster training (larger learning rates possible)
  - Reduces sensitivity to weight initialization
  - Acts as mild regularizer

### 4. Early Stopping

```python
EarlyStopping(
    monitor='val_loss',          # Monitor validation loss
    patience=5,                  # Stop if no improvement for 5 epochs
    restore_best_weights=True    # Restore best model
)
```

- Prevents overfitting by stopping when validation performance plateaus

### 5. Learning Rate Reduction

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                  # Reduce by 50%
    patience=3,                  # After 3 epochs of no improvement
    min_lr=1e-7                  # Minimum learning rate
)
```

- Allows model to fine-tune after plateauing

## Optimizer: Adam

### Why Adam over SGD?

**Adam (Adaptive Moment Estimation)**:
- Adaptive learning rates per parameter
- Good convergence speed
- Robust to hyperparameter choices
- Works well with mini-batches

### Adam Configuration

```python
optimizer = Adam(
    learning_rate=1e-4,
    beta_1=0.9,                  # Exponential decay for first moment
    beta_2=0.999,                # Exponential decay for second moment
    epsilon=1e-7                 # Numerical stability
)
```

### Learning Rate Schedule

- **Phase 1**: 1e-4 (initial training)
  - Fast learning of class-specific patterns
- **Phase 2**: 1e-5 (fine-tuning)
  - 10x slower
  - Preserves pre-trained knowledge
  - Conservative domain adaptation

## Evaluation Metrics Explained

### Per-Class Metrics

For each class:

**Precision** = TP / (TP + FP)
- "Of predicted positives, how many were correct?"
- High precision: Few false positives
- Important when false alarms are costly

**Recall** = TP / (TP + FN)
- "Of actual positives, how many were found?"
- High recall: Few false negatives
- Important when missing positives is costly (medical diagnosis)

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall
- Single metric for class evaluation
- Good for imbalanced classes

**Support** = Total samples in class
- Shows class distribution

### Overall Metrics

**Accuracy** = (TP + TN) / Total
- "Of all predictions, how many were correct?"
- Can be misleading with imbalanced data

**Weighted Average**:
```
weighted_metric = Σ(metric[i] × support[i]) / total_samples
```
- Accounts for class distribution
- Better for imbalanced datasets

### ROC-AUC Curve

**ROC Curve**: Plots True Positive Rate vs False Positive Rate
- Shows model's ability to distinguish at different thresholds
- AUC = Area Under Curve

**AUC Interpretation**:
- 1.0 = Perfect classification
- 0.5 = Random classification
- 0.7-0.8 = Good
- 0.8-0.9 = Very good
- 0.9+ = Excellent

**One-vs-Rest Strategy** (for 10-class problem):
- Generate ROC curve for each class
- Class i (positive) vs all others (negative)
- Average AUC across all classes

## Performance Optimization

### Memory Optimization

**Reduce Batch Size**:
```python
BATCH_SIZE = 16  # Instead of 32
```
- Trade-off: Slower training, less memory
- Better for GPUs with limited VRAM

**Mixed Precision**:
```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```
- Uses float16 (2x faster, less memory)
- Uses float32 where needed (accuracy)

### Speed Optimization

**GPU Acceleration**:
```python
# Verify GPU
tf.config.list_physical_devices('GPU')
```

**Data Prefetching**:
- Already optimized in `DataLoader` using `flow_from_directory`
- Loads next batch while GPU processes current batch

**Gradient Checkpointing**:
```python
# For very large models
# Trade-off: Slower training, less memory
model.compile(..., run_eagerly=True)
```

## Common Issues and Solutions

### Issue 1: Model Overfitting

**Symptoms**:
- Training accuracy: 95%
- Validation accuracy: 70%
- Gap increasing with epochs

**Solutions**:
1. Increase augmentation parameters
2. Increase dropout rates
3. Apply L2 regularization
4. Reduce model capacity (fewer dense neurons)
5. Use early stopping aggressively

### Issue 2: Model Underfitting

**Symptoms**:
- Both training and validation accuracy low (<70%)
- Loss not decreasing

**Solutions**:
1. Increase training epochs
2. Increase learning rate (1e-3 or 5e-4)
3. Reduce regularization (lower dropout, L2)
4. Increase model capacity
5. Check data quality and augmentation

### Issue 3: Class Imbalance Problems

**Symptoms**:
- Model predicts always common class
- Low recall on rare classes

**Solutions**:
1. Enable class weights: `USE_CLASS_WEIGHTS = True`
2. Use focal loss (optional in model.py)
3. Data augmentation focusing on rare classes
4. Adjust class weight scaling

### Issue 4: Training Too Slow

**Symptoms**:
- Each epoch takes >2 minutes

**Solutions**:
1. Enable GPU: `tf.sysconfig.get_build_info()['cuda_version']`
2. Reduce image size: `IMG_SIZE = 224` → `180`
3. Reduce batch size (more updates, less memory)
4. Use mixed precision training
5. Reduce EPOCHS for testing

## Advanced Customization

### Using Different Base Models

```python
# In src/model.py, modify build_model():

# Option 1: EfficientNet (more compact)
base_model = tf.keras.applications.EfficientNetB4(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet'
)

# Option 2: InceptionV3 (different architecture)
base_model = tf.keras.applications.InceptionV3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet'
)

# Option 3: Vision Transformer (state-of-art)
base_model = tf.keras.applications.ViT_B16(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet21k'
)
```

### Custom Loss Functions

```python
# Focal Loss for imbalanced data
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    pt = tf.exp(-ce_loss)
    focal_loss = alpha * tf.pow(1 - pt, gamma) * ce_loss
    return tf.reduce_mean(focal_loss)

# Use in compile:
model.compile(loss=focal_loss, ...)
```

### Ensemble Predictions

```python
models = [
    classifier.load_model('model1.h5'),
    classifier.load_model('model2.h5'),
    classifier.load_model('model3.h5'),
]

# Average predictions
ensemble_pred = np.mean([m.predict(images) for m in models], axis=0)
```

## Dataset Preparation Best Practices

1. **Data Quality**:
   - Remove corrupted/low-quality images
   - Ensure consistent image format (JPG, PNG)
   - Check for duplicate images

2. **Data Labeling**:
   - Verify label accuracy
   - Use inter-rater agreement for quality assurance
   - Document labeling guidelines

3. **Data Balance**:
   - Analyze class distribution
   - Consider data collection strategies for minority classes
   - Document known biases

4. **Privacy**:
   - Remove PHI (Protected Health Information)
   - Anonymize patient IDs
   - Follow HIPAA/GDPR requirements

## Citations and References

1. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015
3. Goodfellow et al., "Deep Learning", MIT Press, 2016
4. Zoph & Quoc, "Neural Architecture Search with Reinforcement Learning", ICLR 2017

## Contact & Support

For issues, questions, or contributions:
- Check README.md for quick start
- Review examples.py for code samples
- Run setup.py for environment verification
