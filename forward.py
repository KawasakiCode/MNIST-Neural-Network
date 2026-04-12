import numpy as np
from activations import ReLU, Softmax

# Forward pass functions
def conv_forward(input_data, filters, biases):
    # input shape: (m, 1, 28, 28)
    # filters shape: (8, 1, 3, 3)

    m = input_data.shape[0]
    num_filters = filters.shape[0]

    pre_activation = np.zeros((m, num_filters, 26, 26))

    for image in range(m):
        for f in range(num_filters):
            for y in range(26):
                for x in range(26):
                    grid = input_data[image, 0, y:y+3, x:x+3]

                    filter_pass = grid * filters[f, 0, :, :]

                    pre_activation[image, f, y, x] = np.sum(filter_pass) + biases[f, 0]
    
    output_data = ReLU(pre_activation)

    cache = (input_data, filters, pre_activation)

    return output_data, cache

# Used to flatten the 4D output of the conv_forward back to 2D array
def flatten_forward(input_data):
    # input data shape: (m, 8, 26, 26)
    m = input_data.shape[0]

    flattened_data = input_data.reshape(m, -1)

    cache = input_data.shape

    return flattened_data, cache

# Forward pass with ReLU
def relu_forward(input_data, weights, biases):
    pre_activation = input_data @ weights + biases
    output_data = ReLU(pre_activation)

    cache = (input_data, weights, pre_activation)

    return output_data, cache

# Forward pass without activation function
def linear_forward(input_data, weights, biases):
    pre_activation = input_data @ weights + biases

    cache = (input_data, weights, pre_activation)

    return pre_activation, cache

# Softmax application
def softmax_forward(input_data):
    probabilities = Softmax(input_data)
    cache = probabilities

    return probabilities, cache