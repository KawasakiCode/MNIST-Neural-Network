import numpy as np

# Create weight and biases
# 3 total layers
# Input layer with 784 input neurons
# 128 neuron hidden layer
# Output layer with 10 output neurons

def initialize_weights_biases(hidden_layer_nodes: int):
    #Initialize the weights not as 0 to allow for change within the network
    W1 = np.random.randn(784, hidden_layer_nodes) * np.sqrt(2.0 / 784)
    W2 = np.random.randn(hidden_layer_nodes, hidden_layer_nodes) * np.sqrt(2.0 / hidden_layer_nodes)
    W3 = np.random.randn(hidden_layer_nodes, 10) * np.sqrt(2.0 / hidden_layer_nodes)

    b1 = np.zeros((1, hidden_layer_nodes))
    b2 = np.zeros((1, hidden_layer_nodes))
    b3 = np.zeros((1, 10))

    return W1, W2, W3, b1, b2, b3

def loss_derivatives(A3: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    # The output error dZ3
    dZ3 = A3 - Y_train
    return dZ3

def output_layer_derivatives(A2: np.ndarray, dZ3: np.ndarray):
    # The output Weights Gradient dW2
    m = A2.shape[0]

    dW3 = (1 / m) * (A2.T @ dZ3)

    # The output Bias Gradient db2
    db3 = (1 / m) * np.sum(dZ3, axis = 0, keepdims = True)

    return dW3, db3

def hidden_layer_derivatives(dZ2: np.ndarray, W2: np.ndarray, Z1: np.ndarray, X_train: np.ndarray):
    m = dZ2.shape[0]

    # The output error after ReLU
    dA1 = dZ2 @ W2.T

    # The output error before ReLU
    relu_derivative = (Z1 > 0)
    dZ1 = dA1 * relu_derivative

    # The output Weights Gradient W1
    dW1 = (1 / m) * (X_train.T @ dZ1)

    # The output Bias Gradient db1
    db1 = (1 / m) * (np.sum(dZ1, axis = 0, keepdims = True))

    return dW1, db1

def second_hidden_layer_derivatives(dZ3: np.ndarray, Z2: np.ndarray, A1: np.ndarray, W3: np.ndarray):
    m = A1.shape[0]

    # The output error after ReLU
    dA2 = dZ3 @ W3.T

    # The output error before ReLU
    relu_derivative = (Z2 > 0)
    dZ2 = dA2 * relu_derivative

    # The outpt Weights Gradient 2
    dW2 = (1 / m) * (A1.T @ dZ2)
    
    # The output Bias Gradient db2
    db2 = (1 / m) * (np.sum(dZ2, axis = 0, keepdims = True))

    return dZ2, dW2, db2



