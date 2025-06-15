import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create output directory
os.makedirs("plots/gan", exist_ok=True)

# Set paths and seed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.gan import GAN
from dataset import load_fashion_mnist

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def plot_loss_history(disc_losses, gen_losses, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss History")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def show_generated_vs_real(generator, real_images, latent_dim, save_path=None):
    #plots images from dataset vs generated images, help from claude
    noise = tf.random.normal([9, latent_dim])
    generated_images = generator(noise, training=False)
    real_sample = real_images[:9]

    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i in range(9):
        axes[i // 3, (i % 3) * 2].imshow(generated_images[i, :, :, 0], cmap="gray")
        axes[i // 3, (i % 3) * 2].set_title("Generated")
        axes[i // 3, (i % 3) * 2].axis("off")

        axes[i // 3, (i % 3) * 2 + 1].imshow(real_sample[i, :, :, 0], cmap="gray")
        axes[i // 3, (i % 3) * 2 + 1].set_title("Real")
        axes[i // 3, (i % 3) * 2 + 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_gan(epochs=100, batch_size=128, latent_dim=64, save_model_path="plots/gan/best_gan.h5"):
    x_train, _, _, _ = load_fashion_mnist(flatten=False, one_hot=False)
    x_train = x_train.astype("float32") / 255.0

    gan = GAN(input_shape=(28, 28, 1), latent_dim=latent_dim, learning_rate=0.0002)
    gan.compile()
    gen_losses = []
    disc_losses = []

    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]

        losses = gan.train_step(real_imgs)
        gen_losses.append(float(losses["gen_loss"]))
        disc_losses.append(float(losses["disc_loss"]))

        if epoch % 5 == 0: ##print out progression
            print(f"Epoch {epoch}/{epochs} | D Loss: {losses['disc_loss']:.4f} | G Loss: {losses['gen_loss']:.4f}")

    gan.generator.save(save_model_path)
    print(f"\ntraning complete. model saved to: {save_model_path}")

    return gan, gen_losses, disc_losses, x_train

if __name__ == "__main__":
    latent_dime = 100
    gan_model, g_losses, d_losses, real_data = train_gan(latent_dim=latent_dime)

    plot_loss_path = "plots/gan/loss_history.png"
    comparison_path = "plots/gan/generated_vs_real.png"

    plot_loss_history(d_losses, g_losses, save_path=plot_loss_path)
    show_generated_vs_real(gan_model.generator, real_data, latent_dim=latent_dime, save_path=comparison_path)

    print("\nFinal Results:")
    print(f"Final generator loss:{g_losses[-1]:.4f}")
    print(f"Final discrminator loss: {d_losses[-1]:.4f}")
    print(f"Loss plot saved to: {plot_loss_path}")
    print(f"image comparison saved to:{comparison_path}")
