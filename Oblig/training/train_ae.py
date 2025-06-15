import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removes some annoying warnings

os.makedirs("plots/ae", exist_ok=True)

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create output directories
os.makedirs("plots/ae", exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import load_fashion_mnist
from Models.autoencoder import build_conv_autoencoder

def show_reconstruction(model, images, n_images=5, save_path=None):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for i in range(n_images):
        plt.subplot(2, n_images, 1 + i)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + i)
        plt.imshow(reconstructions[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def grid_search_ae(x_train, x_valid, learning_rates, epoch_vals, use_dropout=False, save_path="best_ae.h5"):
    best_val_loss = 1e5
    best_model = None
    best_params = {}
    best_history = None

    best_val_loss = 1e5
    best_model = None
    best_params = {}

    if use_dropout:
      save_path="best_ae_dropout.h5"

    for lr in learning_rates:
        for epochs in epoch_vals:
            print(f"Training with lr={lr}, epochs={epochs}, dropout={use_dropout}")
            encoder, decoder, autoencoder = build_conv_autoencoder(use_dropout=use_dropout)
            autoencoder.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=lr))

            start = time.time()
            history = autoencoder.fit(x_train, x_train, validation_data=(x_valid, x_valid), epochs=epochs, batch_size=64, verbose=0)
            end = time.time()

            val_loss = history.history["val_loss"][-1] # last element in loss (might be higher than someplace earlier)
            print(f" val_loss: {val_loss:.4f} (time spent: {end - start:.2f}s)")

            if val_loss < best_val_loss:
                best_model = autoencoder
                save_path 
                best_model.save(save_path)
                best_history = history
                best_params = {"lr": lr, "epochs": epochs, "dropout": use_dropout}
                best_val_loss = val_loss

    print(f"\nBest model: val_loss={best_val_loss:.4f} with params {best_params}")
    return best_model, best_params, best_history

def plot_loss(history, title, lr, save_path=None):
    best_train_loss = min(history.history['loss'])
    best_val_loss = min(history.history['val_loss'])
    plt.figure()
    plt.plot(history.history["loss"], label=f"Train Loss (min: {best_train_loss:.4f}, lr={lr})")
    plt.plot(history.history["val_loss"], label=f"Val Loss (min: {best_val_loss:.4f})")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    x_train_full, x_test, _, _ = load_fashion_mnist(flatten=False, one_hot=False)
    #x_train_full, x_test, _, _ = load_fashion_mnist(flatten=False, one_hot=False, subset_size=5000) # for test
    
    # 10% split val set
    val_split = 0.1
    n_total = len(x_train_full)
    n_train = int(n_total * (1 - val_split))
    split_idx = n_train
    x_train, x_valid = x_train_full[:split_idx], x_train_full[split_idx:]
    """
    lr_vals = [1]
    epoch_vals = [300]
    model_nodrop, params_nodrop, history_nodrop = grid_search_ae(x_train, x_valid, lr_vals, epoch_vals, use_dropout=False)
    plot_loss(history_nodrop, "Best AE Without Dropout", save_path="plots/ae/loss_300epoch.png")
    """
    
    lr_vals = [0.001, 0.01, 0.1, 1.0]
    epoch_vals = [10, 50]
    #epoch_vals = [5]

    print("No dropout")
    model_nodrop, params_nodrop, history_nodrop = grid_search_ae(x_train, x_valid, lr_vals, epoch_vals, use_dropout=False)
    show_reconstruction(model_nodrop, x_valid, n_images=4, save_path="plots/ae/reconstruction_nodropout.png")
    plot_loss(history_nodrop, "Best AE Without Dropout", lr=params_nodrop["lr"], save_path="plots/ae/loss_nodropout.png")

    print("With dropout")
    model_drop, params_drop, history_drop = grid_search_ae(x_train, x_valid, lr_vals, epoch_vals, use_dropout=True)
    show_reconstruction(model_drop, x_valid, n_images=4, save_path="plots/ae/reconstruction_dropout.png")
    plot_loss(history_drop, "Best AE With Dropout", lr=params_drop["lr"],  save_path="plots/ae/loss_dropout.png")

    



        
    """
    results:
    Best model: val_loss=0.2598 with params {'lr': 1.0, 'epochs': 50, 'dropout': False}
    Best model: val_loss=0.2851 with params {'lr': 1.0, 'epochs': 50, 'dropout': True}

    """
