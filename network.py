import numpy as np

# Create weight and biases
# 3 total layers
# Input layer with 784 input neurons
# 128 neuron hidden layer
# Output layer with 10 output neurons

def initialize_weights_biases(hidden_layer_nodes):
    #Initialize the weights not as 0 to allow for change within the network
    W1 = np.random.uniform(-0.5, 0.5, (784, hidden_layer_nodes))
    W2 = np.random.uniform(-0.5, 0.5, (128, 10))

    b1 = np.zeros((hidden_layer_nodes, 1))
    b2 = np.zeros((10, 1))

    return W1, W2, b1, b2
