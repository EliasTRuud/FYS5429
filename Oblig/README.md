## Oblig 

Dataset:

"Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes"
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot


https://www.kaggle.com/datasets/zalando-research/fashionmnist

| Property            | Description                                                                 |
|---------------------|------------------------------------------------------------------------------|
| **Classes**         | 10 clsees of clothing                                                       |
| **Image Size**      | 28 × 28 pixels (grayscale)                                                   |
| **Training Set**    | 60,000 images                                                                |
| **Test Set**        | 10,000 images                                                                |
| **Input Format**    | NumPy arrays                                          |
| **Label Format**    | Integer class labels (0–9)                                                   |



# Fashion-MNIST Models

Small project using CNN, Autoencoder, VAE, and GAN on the Fashion-MNIST dataset.

## How to Run

### Quick test
Run model files directly. This includes basic testing with a subset of samples to test if model works.
```bash
python Models/autoencoder.py
python Models/cnn.py
python Models/vae.py
python Models/gan.py
```

### Full training & plots
Use training scripts. Runs full training with searches. Can easily be modified to extend serach etc.
```bash
python training/train_ae.py
python training/train_cnn.py
python training/train_vae.py
python training/train_gan.py
```

## Output
- Plots saved to `plots/` then a corresponding folder for each model. (e.g plots/cnn)
- Some model weights saved as `.h5` but not used in current code. Right now bit mixed around in main folder and models/

## Dataset
Fashion-MNIST loaded via `dataset.py`, no needed to download.

## Environment
Made use of conda with tensorflow 2.10 on windows. This should hopefully work to create a same env on windows.
Install with:
```bash
conda create --name tf_fashion --file requirements.txt
```

## Notes
Ideally i wanted to create a full main script to run everything then load in models with weights etc, but i got restricted a bit on time. For code i good amount of lecture notes slides/notebooks + some examples found from other people. Then if i got stuck i made use of claude to fix errors or if i had issue with some plotting. At first tried out the QuickDraw sketches, but i was afraid of poor results due to sparse drawings when it came to VAE and GAN especially. As well as the drawings being stored in odd formats (not greyscale images).
