import cupy as np

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

# Functions used to skip python for loops in the convolutional layer to make the process faster
def get_indices(X_shape, f_h, f_w, stride = 1, pad = 0):
    m, c, h, w = X_shape
    out_h = (h + 2 * pad - f_h) // stride + 1
    out_w = (w + 2 * pad - f_w) // stride + 1

    i0 = np.repeat(np.arange(f_h), f_w)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(f_w), f_h * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), f_h * f_w).reshape(-1, 1)

    return k, i, j

def im2col(X, f_h, f_w, stride=1, pad=0):
    p = pad
    X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_indices(X.shape, f_h, f_w, stride, pad)

    # Grab all overlapping patches simultaneously using advanced slicing
    cols = X_padded[:, k, i, j]
    c = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(c * f_h * f_w, -1)
    return cols

def col2im(cols, X_shape, f_h, f_w, stride=1, pad=0):
    # Reconstructs the gradients back into image shapes for backprop
    m, c, h, w = X_shape
    H_padded, W_padded = h + 2 * pad, w + 2 * pad
    X_padded = np.zeros((m, c, H_padded, W_padded))
    k, i, j = get_indices(X_shape, f_h, f_w, stride, pad)

    cols_reshaped = cols.reshape(c * f_h * f_w, -1, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Accumulate gradients for overlapping pixels
    np.add.at(X_padded, (slice(None), k, i, j), cols_reshaped)

    if pad == 0:
        return X_padded
    return X_padded[:, :, pad:-pad, pad:-pad]
