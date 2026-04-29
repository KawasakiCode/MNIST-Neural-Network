import cupy as np
from numpy.activations import ReLU
from numpy.data import load_and_prep_data
import gc

from numpy.forward import convolution_forward_vectorized, flatten_forward, linear_forward, max_pool_forward, relu_forward, softmax_forward

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

Y_raw = np.argmax(Y_test, axis = 1)

saved_data = np.load("trained_weights.npz")

W1 = saved_data['W1']
W2 = saved_data['W2']
W3 = saved_data['W3']
b1 = saved_data['b1']
b2 = saved_data['b2']
b3 = saved_data['b3']
F1 = saved_data['F1']
b_conv = saved_data['b_conv']

W1 = np.asarray(W1)
W2 = np.asarray(W2)
W3 = np.asarray(W3)

b1 = np.asarray(b1)
b2 = np.asarray(b2)
b3 = np.asarray(b3)

F1 = np.asarray(F1)
b_conv = np.asarray(b_conv)

# Free ram
import numpy as old_np
del old_np
gc.collect()

batch_size = 128
num_samples = X_test.shape[0]

print("Test loop started")
total_correct = 0

for i in range(0, num_samples, batch_size):
    X_batch = X_test[i: i+ batch_size]
    Y_batch = Y_test[i: i+ batch_size]
    
    # Pass the data through the convolutional layer
    conv_output_data, conv_cache = convolution_forward_vectorized(X_batch, F1, b_conv, stride = 1, pad = 0)
    conv_relu = ReLU(conv_output_data)
    conv_relu_cache = (conv_relu, F1, conv_output_data)

    max_pool_out,  max_pool_cache = max_pool_forward(conv_relu)

    flattened_data, flatten_cache = flatten_forward(max_pool_out)

    # Pass through first hidden layer
    dense1_output, dense1_cache = relu_forward(flattened_data, W1, b1)

    # Pass through second hidden layer
    dense2_output, dense2_cache = relu_forward(dense1_output, W2, b2)

    # Output layer (Predictions)
    dense3_output, dense3_cache = linear_forward(dense2_output, W3, b3)
    predictions, softmax_cache = softmax_forward(dense3_output)

    # Accuracy Prediction
    final_prediction = np.argmax(predictions, axis = 1)
    true_labels = np.argmax(Y_batch, axis=1)
    total_correct += int(np.sum(final_prediction == true_labels).get())

print(f"Testing finished. Final results: Accuracy: {total_correct/X_test.shape[0] * 100:.2f}%")