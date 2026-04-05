import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases
from activations import ReLU, Softmax
from losses import Categorical_Cross_Entropy

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"
TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)
X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

W1, W2, b1, b2 = initialize_weights_biases(128)

# The intermediate step of the hidden layer which is passed through ReLU
Z1 = X_train @ W1 + b1

# The output of the hidden layer
A1 = ReLU(Z1)

# Pass the output of the hidden layer into the output layer
Z2 = A1 @ W2 + b2

# Final output of the network
A2 = Softmax(Z2)

# Calculate loss
CCE_loss = Categorical_Cross_Entropy(A2, Y_train)





