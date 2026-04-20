# File: check_gpu.py
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Simple computation test
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.random.normal([10000, 10000])
        b = tf.random.normal([10000, 10000])
        c = tf.matmul(a, b)
    print("GPU computation successful!")
else:
    print("No GPU detected ? running on CPU only.")