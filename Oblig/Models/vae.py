import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import load_fashion_mnist

# code pulled from reference code + some changes + claude help with errors when running code i couldnt fix.
class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalLossLayer(layers.Layer):
    def call(self, inputs):
        x_input, z_mean, z_log_var, x_recon = inputs
        reconstruction_loss = tf.reduce_mean(tf.square(x_input - x_recon), axis=[1, 2, 3])
        reconstruction_loss *= 100  # scaling factor
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return x_recon


class CVAE:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=2):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.model = None
        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_model()

    def _build_encoder(self):
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")

        x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
        x = layers.Conv2D(64, 3, activation="relu", padding="same", strides=2)(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = SamplingLayer()([z_mean, z_log_var])

        self.encoder_inputs = inputs
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.z = z
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        conv_shape = (14, 14, 64)  # inferred from Conv2D w/ stride=2
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="z_sampling")
        x = layers.Dense(np.prod(conv_shape), activation="relu")(latent_inputs)
        x = layers.Reshape(conv_shape)(x)

        x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        self.decoder = Model(latent_inputs, x, name="decoder")

    def _build_model(self):
        x_recon = self.decoder(self.z)
        x_out = VariationalLossLayer()([self.encoder_inputs, self.z_mean, self.z_log_var, x_recon])
        self.model = Model(self.encoder_inputs, x_out, name="CVAE")

    def compile(self, lr=1e-3):
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))

    def fit(self, x_train, x_valid, batch_size=32, epochs=10):
        return self.model.fit( x_train, x_train, validation_data=(x_valid, x_valid), epochs=epochs, batch_size=batch_size, verbose=True
        )


if __name__ == "__main__":
    print("Loading Fashion-MNIST...")
    x_train, x_test, _, _ = load_fashion_mnist(flatten=False, one_hot=False, subset_size=5000)

    print("Initializing CVAE...")
    cvae = CVAE(input_shape=(28, 28, 1), latent_dim=2)
    cvae.compile(lr=1e-2)

    print("\nTraining on subset...")
    cvae.fit(x_train, x_test, batch_size=32, epochs=5)
