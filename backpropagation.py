import numpy as np

#Backward pass functions (Backpropagation)
#Grad stands for gradient aka derivative
def backpropagation(grad_output, cache):
    input_data, weights, pre_activation = cache
    m = input_data.shape[0]

    relu_derivative = (pre_activation > 0)
    grad_pre_activation = grad_output  * relu_derivative

    grad_weights = (1 / m) * (input_data.T @ grad_pre_activation)
    grad_biases = (1 / m) * np.sum(grad_pre_activation, axis = 0, keepdims = True)

    grad_input = grad_pre_activation @ weights.T

    return grad_input, grad_weights, grad_biases

def backpropagation_softmax(true_labels, cache):
    probabilities = cache
    m = true_labels.shape[0]

    grad_input = (probabilities - true_labels) / m

    return grad_input