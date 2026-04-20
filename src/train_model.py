import tensorflow as tf
import time

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
print("Num GPUs:", len(gpus))

if not gpus:
    raise RuntimeError("No GPU found – check Slurm allocation and modules")

# Force first GPU (optional)
tf.config.set_visible_devices(gpus[0], 'GPU')

# Quick test: large matrix multiplication on GPU
start = time.time()
with tf.device('/GPU:0'):
    a = tf.random.normal([8000, 8000])
    b = tf.random.normal([8000, 8000])
    c = tf.matmul(a, b)
print(f"GPU matmul took {time.time() - start:.2f} seconds")

# Your real training code here...
print("Training would start now...")