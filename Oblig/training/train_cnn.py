import os
os.makedirs("plots/cnn", exist_ok=True)
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#allows import from further up
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

from dataset import load_fashion_mnist
from Models.cnn import CNN_model

# fix randomness for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# seaborn/matplotlib aesthetics
sns.set_style("darkgrid")
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)


def grid_search_cnn(x_train, y_train, x_test, y_test, eta_vals, lmbd_vals, epochs=10, batch_size=100, save_path="best_cnn.h5"):
    """
    grid searches for learning rate and regularization lambda, then saves best model
    """
    train_acc = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_acc = np.zeros((len(eta_vals), len(lmbd_vals)))
    best_score = 0
    best_model = None

    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            print(f"Training model: eta={eta}, lambda={lmbd}")
            model = CNN_model(input_shape= x_train.shape[1:], eta=eta, lmbd=lmbd)
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            train_score = model.evaluate(x_train, y_train, verbose=0)[1]
            test_score = model.evaluate(x_test, y_test, verbose=0)[1]

            train_acc[i, j] = train_score
            test_acc[i, j] = test_score

            if test_score > best_score:
                best_score = test_score
                best_model = model
                model.save(save_path)
                best_eta = eta
                best_lmbd = lmbd

            print(f"Test accuracy: {test_score:.4f}\n")

    return train_acc, test_acc, best_score, best_eta, best_lmbd


def plot_heatmap(data, x_labels, y_labels, title="Accuracy", show=False):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, annot=True, fmt=".3f", cmap="viridis", ax=ax) #gpt help
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Eta")
    ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_fashion_mnist(flatten=False, one_hot=True, subset_size=5000)

    epochs_list = [10, 50, 100]
    batch_sizes = [64, 128]

    #epochs_list = [5, 10, 50] #test out some more values
    #batch_sizes = [16, 32]

    eta_vals = np.logspace(-5, -1, 5)
    lmbd_vals = np.logspace(-5, -1, 5)

    # Did some test of epochs and batch size: results. Dont need to run
    """
    
    for epochs_ in epochs_list:
        for batch_size_ in batch_sizes:
            start = time.time()
            train_acc, test_acc, best, best_eta, best_lmbd = grid_search_cnn( x_train, y_train, x_test, y_test, eta_vals=eta_vals, lmbd_vals=lmbd_vals, epochs=epochs_, batch_size=batch_size_, save_path="best_cnn.h5")
            end = time.time()
            
           
            print(f"For epochs: {epochs_} and batch size: {batch_size_}")
            print(f"Best test accuracy: {best:.4f} with lambda: {best_lmbd:.2e} and eta: {best_eta:.2e}")
            print(f"Time spent training: {(end - start):.2f} seconds\n")

             # Plot and save heatmap
            fig = plot_heatmap(test_acc, lmbd_vals, eta_vals)
            fig.savefig(f"plots/cnn/cnn_epochs{epochs_}_bs{batch_size_}.png")
            plt.close(fig)

    """
    

    # Good default config epochs=10, batch=32, eta=0.1, lambda=0.857. Test acc =  0.8566. time = 160s
    # Not as good as epoch=100, batch = 64 (acc = 0.8668), but a lot faster. time = 1127
    best_batch = 32
    best_epoch = 10
    train_acc, test_acc, best, best_eta, best_lmbd = grid_search_cnn( x_train, y_train, x_test, y_test, eta_vals=eta_vals, lmbd_vals=lmbd_vals, epochs=best_epoch, batch_size=best_batch, save_path="best_cnn.h5")
    plot_heatmap(test_acc, lmbd_vals, eta_vals, title="Test Accuracy", show=True)

"""
For epochs: 10 and batch size: 64
Best test accuracy: 0.8380 with lambda: 1.00e-05 and eta: 1.00e-01
Time spent training: 131.13 seconds

For epochs: 10 and batch size: 128
Best test accuracy: 0.8008 with lambda: 1.00e-05 and eta: 1.00e-01
Time spent training: 122.06 seconds

For epochs: 50 and batch size: 64
Best test accuracy: 0.8588 with lambda: 1.00e-04 and eta: 1.00e-01
Time spent training: 571.86 seconds

For epochs: 50 and batch size: 128
Best test accuracy: 0.8108 with lambda: 1.00e-04 and eta: 1.00e-01
Time spent training: 520.26 seconds

For epochs: 100 and batch size: 64
Best test accuracy: 0.8668 with lambda: 1.00e-04 and eta: 1.00e-01
Time spent training: 1127.24 seconds

For epochs: 100 and batch size: 128
Best test accuracy: 0.8502 with lambda: 1.00e-05 and eta: 1.00e-01
Time spent training: 1004.37 seconds

---------------------------------------------------


For epochs: 5 and batch size: 16
Best test accuracy: 0.7894 with lambda: 1.00e-05 and eta: 1.00e-01
Time spent training: 124.87 seconds

For epochs: 5 and batch size: 32
Best test accuracy: 0.7018 with lambda: 1.00e-03 and eta: 1.00e-02
Time spent training: 94.08 seconds

For epochs: 10 and batch size: 16
Best test accuracy: 0.8540 with lambda: 1.00e-04 and eta: 1.00e-01
Time spent training: 226.85 seconds

For epochs: 10 and batch size: 32
Best test accuracy: 0.8566 with lambda: 1.00e-05 and eta: 1.00e-01
Time spent training: 160.71 seconds

For epochs: 50 and batch size: 16
Best test accuracy: 0.8652 with lambda: 1.00e-04 and eta: 1.00e-01
Time spent training: 1028.74 seconds

For epochs: 50 and batch size: 32
Best test accuracy: 0.8610 with lambda: 1.00e-03 and eta: 1.00e-01
Time spent training: 753.53 seconds

"""