import cupy as np

# Backward pass functions (Backpropagation)
# Grad stands for gradient aka derivative

#Backpropagate for the convolutional layer
def backpropagation_conv(grad_output, cache):
    input_data, filters, pre_activation = cache

    m = input_data.shape[0]
    num_filters = filters.shape[0]

    relu_derivative = (pre_activation > 0)
    grad_pre_activation = grad_output * relu_derivative

    grad_filters = np.zeros((8, 1, 3, 3))
    grad_biases = np.zeros((8, 1))

    for image in range(m):
        for filter in range(num_filters):
            grad_biases[filter, 0] += np.sum(grad_pre_activation[image, filter, :, :])
            for y in range(26):
                for x in range(26):
                    grid = input_data[image, 0, y:y+3, x:x+3]
                    grad_filters[filter, 0, :, :] = grid * grad_pre_activation[image, filter, y, x]
    
    return grad_filters, grad_biases

# Backpropagate to unflatten the gradient of the first layer
def backpropagation_unflatten(grad_output, cache):
    # Grad output is the flat array from dense 1
    # cache contains the original shape of the layer before flattening
    original_shape = cache

    grad_input = grad_output.reshape(original_shape)

    return grad_input

# Backpropagate with relu activation function
def backpropagation_relu(grad_output, cache):
    input_data, weights, pre_activation = cache
    m = input_data.shape[0]

    relu_derivative = (pre_activation > 0)
    grad_pre_activation = grad_output  * relu_derivative

    grad_weights = (1 / m) * (input_data.T @ grad_pre_activation)
    grad_biases = (1 / m) * np.sum(grad_pre_activation, axis = 0, keepdims = True)

    grad_input = grad_pre_activation @ weights.T

    return grad_input, grad_weights, grad_biases

# Backpropagate with no activation function
def linear_backward(grad_output, cache):
    input_data, weights, _ = cache
    m = input_data.shape[0]

    grad_weights = (1 / m) * (input_data.T @ grad_output)
    grad_biases = (1 / m) * np.sum(grad_output, axis = 0, keepdims= True)

    grad_input = grad_output @ weights.T

    return grad_input, grad_weights, grad_biases

# Backpropagate with CCE loss and Softmax into a single equation
def backpropagation_softmax(true_labels, cache):
    probabilities = cache
    m = true_labels.shape[0]

    grad_input = (probabilities - true_labels) / m

    return grad_input

def backpropagation_vectorized(dout, cache):
    X, W, b, stride, pad, X_col = cache
    num_filters, _, f_h, f_w = W.shape

    # 1. Gradients for Biases
    db = np.sum(dout, axis=(0, 2, 3)).reshape(num_filters, -1)

    # 2. Gradients for Weights (Filters)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    return dW, db

def max_pool_backpropagation(grad_output, cache):
    mask = cache

    height_grad_unflatten = np.repeat(grad_output, 2, axis = 2)
    width_grad_unflatten = np.repeat(height_grad_unflatten, 2, axis = 3)

    output = mask * width_grad_unflatten

    return output

def relu_conv_backward(grad_output, cache):
    input_data, weights, pre_activation = cache

    mask = pre_activation > 0
    grad = grad_output * mask

    return grad

