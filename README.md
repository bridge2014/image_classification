# image_classification
 A python project  to classify the medical image using tensorflow ResNet50 model. The train and test datasets are available, each one has 10 classes. 

First run validate_images.py to find out any corrupted image files and then remove them from folder.
I do find out one image file which is corrupted.

Total 72101 files belonging to 10 classes in train folder. 
Using 61286 files for training.
Using 10815 files for validation.

Total 18030 files belonging to 10 classes in test folder.

All computation was done in Stony Brook University high performance linux cluster. 
The init.sh was executed first to create virtural environment, then create job.slurm file
to use GPU . All results are captured in png files and classification report. 
The prediction accuracy for training data is over 97% and it close to 95% for validation dataset.
The prediction accuracy for test data is around 95%.

There are 2 way to wrap image with tensorflow keras module

Method 1: 

import tensorflow as tf

train_dir = "/path/to/train_folder"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,   # 20% goes to validation
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

Method 2:
This method is Older Keras Example (ImageDataGenerator) Still works, but less recommended today;

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset="training",
    seed=123
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    subset="validation",
    seed=123
)
As I used both methods, Method 2 with much high prediction accuracy.


