import numpy as np

# Create weight and biases
# 3 total layers
# Input layer with 784 input neurons
# 128 neuron hidden layer
# Output layer with 10 output neurons

def initialize_weights_biases(hidden_layer_nodes: int):
    #Initialize the weights not as 0 to allow for change within the network
    W1 = np.random.uniform(-0.5, 0.5, (784, hidden_layer_nodes))
    W2 = np.random.uniform(-0.5, 0.5, (hidden_layer_nodes, 10))

    b1 = np.zeros((1, hidden_layer_nodes))
    b2 = np.zeros((1, 10))

    return W1, W2, b1, b2

def loss_derivatives(A2: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    #The output error dZ2
    dZ2 = A2 - Y_train
    return dZ2

def output_layer_derivatives(A1: np.ndarray, dZ2: np.ndarray):
    #The output Weights Gradient dW2
    m = A1.shape[0]

    dW2 = (1 / m) * (A1.T @ dZ2)

    #The output Bias Gradient db2
    db2 = (1 / m) * np.sum(dZ2, axis = 0, keepdims = True)

    return dW2, db2

def hidden_layer_derivatives(dZ2: np.ndarray, W2: np.ndarray, Z1: np.ndarray, X_train: np.ndarray):
    m = dZ2.shape[0]

    #The output error after ReLU
    dA1 = dZ2 @ W2.T

    #The output error before ReLU
    relu_derivative = (Z1 > 0)
    dZ1 = dA1 * relu_derivative

    #The output Weights Gradient W1
    dW1 = (1 / m) * (X_train.T @ dZ1)

    #The output Bias Gradient db1
    db1 = (1 / m) * (np.sum(dZ1, axis = 0, keepdims = True))

    return dW1, db1


