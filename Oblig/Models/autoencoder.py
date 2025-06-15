import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from dataset import load_fashion_mnist

def build_conv_autoencoder(use_dropout=False):
    """
    makes a convolutional autoencoder with encoder+decoder
    args:
        use_dropout (bool): if True, adds dropout layers in encoder
    returns: encoder, decoder, autoencoder as keras models
    """
    encoder_layers = [
        layers.Reshape([28, 28, 1], input_shape=[28, 28]),
        layers.Conv2D(16, kernel_size=3, padding="same", activation="selu")
    ]
    if use_dropout:
        encoder_layers.append(layers.Dropout(0.2))
    encoder_layers.append(layers.MaxPooling2D(pool_size=2))

    encoder_layers.append(layers.Conv2D(32, kernel_size=3, padding="same", activation="selu"))
    if use_dropout:
        encoder_layers.append(layers.Dropout(0.2))
    encoder_layers.append(layers.MaxPooling2D(pool_size=2))

    encoder_layers.append(layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"))
    if use_dropout:
        encoder_layers.append(layers.Dropout(0.3))
    encoder_layers.append(layers.MaxPooling2D(pool_size=2))

    encoder = models.Sequential(encoder_layers)

    decoder = models.Sequential([
        layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="valid", activation="selu", input_shape=[3, 3, 64]),
        layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="same", activation="selu"),
        layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid"),
        layers.Reshape([28, 28])
    ])

    autoencoder = models.Sequential([encoder, decoder])
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    # test run
    x_train, x_test, _, _ = load_fashion_mnist(flatten=False, one_hot=False, subset_size=1000)

    encoder, decoder, autoencoder = build_conv_autoencoder(use_dropout=True)
    autoencoder.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=1.0))

    autoencoder.summary()
    autoencoder.fit(x_train, x_train, epochs=3, batch_size=64, validation_split=0.1)