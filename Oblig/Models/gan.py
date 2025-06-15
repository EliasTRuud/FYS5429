import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import load_fashion_mnist


class GAN(tf.keras.Model):
    # Full class of GAN with a generator and discriminator. Architecture mostly pulled from example code. Help from claude for the trianing  step and loss.
    def __init__(self, input_shape=(28, 28, 1), latent_dim=64, learning_rate=1e-4):
        super(GAN, self).__init__()
        self.img_shape = input_shape
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def build_generator(self):
        model = models.Sequential(name="Generator")
        model.add(layers.Input(shape=(self.latent_dim,)))
        model.add(layers.Dense(7 * 7 * 64, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Reshape((7, 7, 64)))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid'))
        return model


    def build_discriminator(self):
        model = models.Sequential(name="Discriminator")
        model.add(layers.Input(shape=self.img_shape))
        model.add(layers.Conv2D(32, 3, strides=2, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(64, 3, strides=2, padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        return model

    def compile(self):
        super(GAN, self).compile()
        # losses and optimizers are already assigned in init

    def gan_loss(self, logits, is_real):
        labels = tf.ones_like(logits) if is_real else tf.zeros_like(logits)
        return self.cross_entropy(labels, logits)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Get discriminator predictions
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate losses
            gen_loss = self.gan_loss(fake_output, is_real=True)
            disc_loss = self.gan_loss(real_output, is_real=True) + self.gan_loss(fake_output, is_real=False)

        # Compute gradients
        gen_gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return {"gen_loss": gen_loss, "disc_loss": disc_loss}

    def generate_images(self, n=10):
        noise = tf.random.normal([n, self.latent_dim])
        return self.generator(noise, training=False)


if __name__ == "__main__":
    x_train, _, _, _ = load_fashion_mnist(flatten=False, one_hot=False, subset_size=1024)

    gan = GAN()
    gan.compile()

    print("\nTraining GAN for 3 epochs on small dataset...")

    for epoch in range(3):
        print(f"Epoch {epoch+1}")
        for i in range(0, len(x_train), 32):
            batch = x_train[i:i+32]
            gan.train_step(batch)
    print("Training complete.")



"""
had issue with checkboard artifacting:

Checkerboard artifacts arise from using Conv2DTranspose layers (i.e., transposed convolutions or "deconvs") for upsampling.

These artifacts are:

visible before training (initial weights produce patterns).

present during/after training, distorting generation quality.

Solution: Replace transposed convolutions with resize-then-convolve:

Upsample using tf.image.resize(), then apply a normal Conv2D.

instad of 
model.add(layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False))

Use this:
model.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
model.add(layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False))

"""