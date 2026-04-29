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
    #With max pool added W1 gets 1352 so its of shape (64, 1352) per batch
    input_nodes = 1352
    W1 = np.random.randn(input_nodes, hidden_layer_nodes) * np.sqrt(2.0 / input_nodes)
    W2 = np.random.randn(hidden_layer_nodes, hidden_layer_nodes) * np.sqrt(2.0 / hidden_layer_nodes)
    W3 = np.random.randn(hidden_layer_nodes, 10) * np.sqrt(2.0 / hidden_layer_nodes)

    b1 = np.zeros((1, hidden_layer_nodes))
    b2 = np.zeros((1, hidden_layer_nodes))
    b3 = np.zeros((1, 10))

    # Adam m Matrices (Exponentially Weighted Moving Average)
    adam_W1_m = np.zeros((input_nodes, hidden_layer_nodes))
    adam_W2_m = np.zeros((hidden_layer_nodes, hidden_layer_nodes))
    adam_W3_m = np.zeros((hidden_layer_nodes, 10))

    adam_b1_m = np.zeros((1, hidden_layer_nodes))
    adam_b2_m = np.zeros((1, hidden_layer_nodes))
    adam_b3_m = np.zeros((1, 10))

    # Adam u Matrices (Exponentially Weighted Moving Average squared)
    adam_W1_u = np.zeros((input_nodes, hidden_layer_nodes))
    adam_W2_u = np.zeros((hidden_layer_nodes, hidden_layer_nodes))
    adam_W3_u = np.zeros((hidden_layer_nodes, 10))

    adam_b1_u = np.zeros((1, hidden_layer_nodes))
    adam_b2_u = np.zeros((1, hidden_layer_nodes))
    adam_b3_u = np.zeros((1, 10))

    adam_m = [adam_W1_m, adam_W2_m, adam_W3_m, adam_b1_m, adam_b2_m, adam_b3_m]
    adam_u = [adam_W1_u, adam_W2_u, adam_W3_u, adam_b1_u, adam_b2_u, adam_b3_u]

    return W1, W2, W3, b1, b2, b3, adam_m, adam_u

def initialize_cnn_filters(num_filters: int):
    #F1 refers to filter 1 and replaces W1
    F1 = np.random.randn(num_filters, 1, 3, 3) * np.sqrt(2.0 / (3 * 3))

    #One bias number per filter so 8 bias
    b_conv = np.zeros((num_filters, 1))

    adam_F1_m = np.zeros((num_filters, 1, 3, 3))
    adam_b_conv_m = np.zeros((num_filters, 1))

    adam_F1_u = np.zeros((num_filters, 1, 3, 3))
    adam_b_conv_u = np.zeros((num_filters, 1))

    adam_m = [adam_F1_m, adam_b_conv_m]
    adam_u = [adam_F1_u, adam_b_conv_u]

    return F1, b_conv, adam_m, adam_u

# Functions used to skip python for loops in the convolutional layer to make the process faster
# Stride is the step of the filter. In this case the filter moves 1 pixel to the right every time 
# Filter now sees only 26x26 pixels of the image so it doesnt hang off the edge
# By making pad = 1 we make this 28x28 so it sees the whole image
def get_indices(X_shape, f_h, f_w, stride = 1, pad = 0):
    m, c, h, w = X_shape
    out_h = (h + 2 * pad - f_h) // stride + 1
    out_w = (w + 2 * pad - f_w) // stride + 1

    i0 = np.repeat(np.arange(f_h), f_w)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(f_w), f_h * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    # i is a 9x676 array containing all row coordinates for every pixel of each one of 
    # the possible 676 filters. So i[0][0] would be the row coordinate of the top left pixel
    # of the first possible filter. j is the same but for columns
    # k is the channel (depth) coordinate. In this case its a 9x1 matrix with 9 zeros
    # But in a colored photo with RGB there would be 3 channels and k would be a 27x1
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), f_h * f_w).reshape(-1, 1)

    return k, i, j

def im2col(X, f_h, f_w, stride=1, pad=0):
    p = pad
    X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_indices(X.shape, f_h, f_w, stride, pad)

    # Grab all overlapping patches simultaneously using advanced slicing
    # cols here is essentialy all the filters of all the images of the batch
    # In the end cols is of shape (64, 9, 676)
    # .transpose makes it a (9, 676, 64) shape 
    # and .reshape makes is a (9, 43264)
    cols = X_padded[:, k, i, j]
    c = X.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(c * f_h * f_w, -1)
    return cols

# col2im is only used in backpropagation to calculate the derivative of the layer
# since the convolutional layer in this network is first we don't need that derivative so 
# we don't use it
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
