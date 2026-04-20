import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import random


message="absl:Compiled the loaded model";
# Just ignore / silence the warning
warnings.filterwarnings(
    "ignore",
    message=message,
    category=Warning
)

warnings.filterwarnings(
    "ignore",
    message="Your `PyDataset` class should call",
    category=UserWarning
)


#  Inference on a single image
print(" --- Inference on a single image ----")
def predict_cancer_type(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return class_names[class_idx], prediction[0][class_idx]


def get_random_files(root_dir: str, count: int = 10) -> list[Path]:
    """
    Randomly select 'count' files from root_dir and all its subfolders.
    """
    root = Path(root_dir)
    
    # Get all files recursively (excluding directories and hidden files)
    all_files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith('.')]
    
    if not all_files:
        print("No files found.")
        return []
    
    if len(all_files) < count:
        print(f"Only {len(all_files)} files available.")
        return all_files
    
    return random.sample(all_files, count)
    
    

project_base_path="/vast/home/fwang/image_ai/"
model_path = os.path.join(project_base_path, 'results/resnet50_cancer_classifier3.h5') 

img_height, img_width = 224, 224  # ResNet50 input size

test_dir = '/vast/home/fwang/image_ai/data/test/'
class_names = sorted(os.listdir(test_dir))  # e.g., ['class_0', ..., 'class_9']

files = get_random_files(test_dir, 10)
print(f"\nSelected {len(files)} random files:")

for i, file_name in enumerate(files, 1):
  print(f"{file_name}")
  #full_path = os.path.join(test_dir, file_name);
  #print(f"image file path : {full_path}")       
  cancer_type, confidence = predict_cancer_type(file_name,model_path)
  print(f"cancer_type is {cancer_type} and confidence is : {confidence}")
  print ('------------------------------')
