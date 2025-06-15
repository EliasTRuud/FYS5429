import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import load_fashion_mnist

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, optimizers, regularizers


def CNN_model(input_shape, receptive_field=3, n_filters=16, n_neurons=64, n_classes=10, eta=0.01, lmbd=0.001):
    """
    Simple CNN based on lecture notes. One conv + dense layer, regularized.
    Args:
        input_shape (tuple): Input image shape (H, W, C)
        receptive_field (int): Size of conv kernel (3 = 3x3)
        n_filters (int): Number of conv filters
        n_neurons (int): Hidden layer neurons
        n_classes (int): Output layer size
        eta (float): Learning rate
        lmbd (float): L2 regularization strength

    Returns:
        model, to be used on new data
    """
    model = Sequential()
    model.add(layers.Conv2D(n_filters, (receptive_field, receptive_field), activation="relu", padding="same", kernel_regularizer=regularizers.l2(lmbd), input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_neurons, activation="relu", kernel_regularizer=regularizers.l2(lmbd)))
    model.add(layers.Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(lmbd)))

    sgd = optimizers.SGD(learning_rate=eta)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # small quick test
    from dataset import load_fashion_mnist
    x_train, x_test, y_train, y_test = load_fashion_mnist(flatten=False, one_hot=True, subset_size=2000)

    model = CNN_model(input_shape=x_train.shape[1:])
    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)
    print("Test acc:", model.evaluate(x_test, y_test)[1])
