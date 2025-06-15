import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# set seed, for const rand
np.random.seed(42)
tf.random.set_seed(42)

#seaborn params for plot
colors = sns.color_palette("deep")
sns.set_style('darkgrid')
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=12)
plt.rc('font', size=12)

def load_fashion_mnist(normalize=True, flatten=False, one_hot=False, subset_size=None):
    """
    load the train and test dataset with some arguments depending on whats needed
    args:
        normalize(bool): scale from 0 to 1
        flatten (bool):flattn to 1d
        one_hot (bool): one hod encodes
        subset_size (int): trim set for quick tests if needed

    return:
        x_train, x_test, y_train, y_test: numpy arrys
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    if subset_size:
        x_train, y_train = x_train[:subset_size], y_train[:subset_size]
        x_test, y_test = x_test[:subset_size], y_test[:subset_size]

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if not flatten:
        x_train = x_train[..., np.newaxis]  # add dim
        x_test = x_test[..., np.newaxis]
    else:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test

def show_random_image(x_data, y_data, class_names=None):
    """
    show one plot of a random img from the set, gen from seed (42). GPT help, had issue with dimS

    Args:
        x_data (np.ndarray): Image data (either with or without channel)
        y_data (np.ndarray): Labels (int or one-hot)
        class_names (list): Optional list of label names
    """
    idx = np.random.randint(0, len(x_data))
    img = x_data[idx]
    label = y_data[idx]

    if img.ndim == 3:
        img = img.squeeze()  # remchannel if dims there

    if label.ndim > 0 and label.shape[-1] == 10:
        label = np.argmax(label)

    label_str = str(label)
    if class_names and isinstance(class_names, list):
        label_str = class_names[label]

    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label_str}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    x_train, x_test, y_train, y_test = load_fashion_mnist(flatten=False, one_hot=False, subset_size=5000)
    print("Train shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    show_random_image(x_train, y_train, class_names)
