import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPUs Disponibles:", tf.config.list_physical_devices('GPU'))

# Test de operación en GPU
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    print("Resultado GPU:", a + b)