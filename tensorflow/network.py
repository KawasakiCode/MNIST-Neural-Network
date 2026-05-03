import tensorflow as tf
import keras
from keras import Model

class MNIST(Model):
    def __init__(self):
        super().__init__()
        
        # Convolutional layer
        # 1 color channel
        # 8 filters
        # 3x3 filter size
        self.conv = keras.layers.Conv2D(filters=8, kernel_size=3)

        # ReLU
        self.relu = keras.layers.ReLU()

        # Max Pool layer
        # 2x2 block size
        # Stride 2 (match block size)
        self.maxpool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        # Dense Layers
        # Flatten Max Pool output
        self.flatten = keras.layers.Flatten()

        # Hidden Layer 1
        self.dense1 = keras.layers.Dense(units=128)

        # Dropout Regularization
        self.dropout = keras.layers.Dropout(rate=0.2)

        # Hidden Layer 2
        self.dense2 = keras.layers.Dense(units=10)

    def call(self, inputs, training=False):
        # Forward Pass function
        # TensorFlow automatically stores cache

        x = self.conv(inputs)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x, training=training)

        logits = self.dense2(x)

        return logits