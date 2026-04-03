import numpy as np

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"
TEST_FILEPATH = "mnist_test/mnist_test.csv"

def load_and_prep_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    #First column (labels)
    Y_raw = data[:, 0]

    #Pixel data (28x28 = 784 values)
    X_raw = data[:, 1:]

    #Normalize values so they are between 0.0 and 1.0
    #Neural networks work faster with smaller numbers
    X = X_raw / 255.0

    #Create an array of zeros (number of images x 10)
    num_images = Y_raw.shape[0]
    Y_encoded = np.zeros((num_images, 10))

    Y_encoded[np.arange(num_images), Y_raw.astype(int)] = 1.0

    return X, Y_encoded