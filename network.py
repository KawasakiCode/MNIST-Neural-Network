import numpy as np

# Create weight and biases
# 3 total layers
# Input layer with 784 input neurons
# 128 neuron hidden layer
# Output layer with 10 output neurons
def initialize_weights_biases(hidden_layer_nodes: int):
    #Initialize the weights not as 0 to allow for change within the network
    #Use 5408 and not 784 because W1 wont interact with the original
    #pixel values but the the flatten_output of the convolutional
    #layer which is of shape (60000, 5408)
    W1 = np.random.randn(5408, hidden_layer_nodes) * np.sqrt(2.0 / 5408)
    W2 = np.random.randn(hidden_layer_nodes, hidden_layer_nodes) * np.sqrt(2.0 / hidden_layer_nodes)
    W3 = np.random.randn(hidden_layer_nodes, 10) * np.sqrt(2.0 / hidden_layer_nodes)

    b1 = np.zeros((1, hidden_layer_nodes))
    b2 = np.zeros((1, hidden_layer_nodes))
    b3 = np.zeros((1, 10))

    return W1, W2, W3, b1, b2, b3

def initialize_cnn_filters(num_filters: int):
    #F1 refers to filter 1 and replaces W1
    F1 = np.random.randn(num_filters, 1, 3, 3) * np.sqrt(2.0 / (3 * 3))

    #One bias number per filter so 8 bias
    b_conv = np.zeros((num_filters, 1))

    return F1, b_conv



