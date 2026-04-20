import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

print("GPUs:", gpus)

# Simple test computation on GPU
if gpus:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("Computation result on GPU:\n", c)
else:
    print("No GPU detected by TensorFlow")