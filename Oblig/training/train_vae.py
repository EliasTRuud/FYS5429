import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes some annoying warnings

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from Models.vae import CVAE
from dataset import load_fashion_mnist


def make_dirs():
    os.makedirs("plots/vae", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def plot_loss(history, title, save_path):
    # PLotting the loss function, claude help.
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Val Loss")

    min_train = np.min(history.history["loss"])
    text = f"Min Train Loss: {min_train:.4f}"
    if "val_loss" in history.history:
        min_val = np.min(history.history["val_loss"])
        text += f", Min Val Loss: {min_val:.4f}"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_reconstruction(model, x, save_path):
    # claude help
    import matplotlib.pyplot as plt

    preds = model.decoder.predict(model.encoder.predict(x)[2])

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axes[0, i].imshow(x[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(preds[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def grid_search_vae(x_train, x_valid, lr_vals, epoch_vals, latent_dim=2):
    best_loss = float("inf")
    best_model = None
    best_params = None
    best_history = None

    for lr in lr_vals:
        for epochs in epoch_vals:
            print(f"Training VAE with lr={lr}, epochs={epochs}")

            model = CVAE(input_shape=(28, 28, 1), latent_dim=latent_dim)
            model.compile(lr=lr)

            early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            history = model.model.fit( x_train, x_train, validation_data=(x_valid, x_valid), epochs=epochs, batch_size=64, callbacks=[early_stop], verbose=0,)

            final_val_loss = history.history["val_loss"][-1]
            if final_val_loss < best_loss:
                best_loss = final_val_loss
                best_model = model
                best_params = {"lr": lr, "epochs": epochs}
                best_history = history
                model.model.save("models/best_vae.h5")
                print(f"New best model saved with val_loss={best_loss:.4f}")

    return best_model, best_params, best_history


if __name__ == "__main__":
    make_dirs()

    x_train_full, x_test, _, _ = load_fashion_mnist(flatten=False, one_hot=False)
    val_split = 0.1
    split_idx = int(len(x_train_full) * (1 - val_split))
    x_train, x_valid = x_train_full[:split_idx], x_train_full[split_idx:]

    lr_vals = [1e-4, 1e-3, 1e-2]
    lr_vals = [1e-4, 1e-3]
    epoch_vals = [20, 50, 100]

    #lr_vals = [0.01]
    #epoch_vals = [2]

    model, params, history = grid_search_vae(x_train, x_valid, lr_vals, epoch_vals, latent_dim=10) #set latent dim to 10, since 10 categories

    final_loss = history.history['val_loss'][-1]
    print(f"Best params: {params}, Final validation loss: {final_loss:.4f}")

    plot_loss(history, "VAE Loss", "plots/vae/loss.png")
    save_reconstruction(model, x_test, "plots/vae/reconstruction.png")
