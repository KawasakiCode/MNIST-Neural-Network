import cupy as np
from numpy.activations import ReLU
from numpy.data import augment_data, load_and_prep_data
from numpy.network import initialize_weights_biases, initialize_cnn_filters
from numpy.forward import convolution_forward_vectorized, linear_forward, max_pool_forward, relu_forward, flatten_forward, softmax_forward
from numpy.backpropagation import backpropagation_relu, backpropagation_softmax, backpropagation_unflatten, backpropagation_vectorized, linear_backward, max_pool_backpropagation, relu_conv_backward
from numpy.losses import Categorical_Cross_Entropy
from numpy.optimizers import Adam, SGD

import gc

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)

# Free ram
import numpy as old_np
del old_np
gc.collect()

Y_raw = np.argmax(Y_train, axis = 1)

batch_size = 64
num_samples = X_train.shape[0]

W1, W2, W3, b1, b2, b3, adam_m_dense, adam_u_dense = initialize_weights_biases(128)
F1, b_conv, adam_m_conv, adam_u_conv = initialize_cnn_filters(8)

adam_m = adam_m_dense + adam_m_conv
adam_u = adam_u_dense + adam_u_conv

W1 = np.asarray(W1)
W2 = np.asarray(W2)
W3 = np.asarray(W3)

b1 = np.asarray(b1)
b2 = np.asarray(b2)
b3 = np.asarray(b3)

F1 = np.asarray(F1)
b_conv = np.asarray(b_conv)

# Lower lr cause adam accelerates the weights by itself
lr = 0.001
# First Moment Decay
b1_momentum = 0.9
# Second Moment Decay
b2_scaling = 0.999
# Very small number to avoid division by 0
e = 1e-8

t = 0

print("Training loop started")
for epoch in range(30):
    permutation = np.random.permutation(num_samples)
    X_train_shuffled = X_train[permutation]
    Y_train_shuffled = Y_train[permutation]
    
    epoch_loss = 0
    epoch_accuracy = 0
    batch_in_epoch = 0


    for i in range(0, num_samples, batch_size):
        X_batch = X_train_shuffled[i : i + batch_size]
        Y_batch = Y_train_shuffled[i : i + batch_size]
        
        # Data augmentation (Translation: pixel shifting)
        X_batch = augment_data(X_batch)
        
        # Pass the data through the convolutional layer
        conv_output_data, conv_cache = convolution_forward_vectorized(X_batch, F1, b_conv, stride = 1, pad = 0)
        conv_relu = ReLU(conv_output_data)
        conv_relu_cache = (conv_relu, F1, conv_output_data)

        max_pool_out,  max_pool_cache = max_pool_forward(conv_relu)

        flattened_data, flatten_cache = flatten_forward(max_pool_out)

        # Pass through first hidden layer
        dense1_output, dense1_cache = relu_forward(flattened_data, W1, b1)

        random_matrix_dense1 = np.random.rand(*dense1_output.shape)
        mask_dense1 = random_matrix_dense1 < 0.8

        new_dense1_output = (dense1_output * mask_dense1) / 0.8

        # Pass through second hidden layer
        dense2_output, dense2_cache = relu_forward(new_dense1_output, W2, b2)

        # Output layer (Predictions)
        dense3_output, dense3_cache = linear_forward(dense2_output, W3, b3)
        predictions, softmax_cache = softmax_forward(dense3_output)

        # Accuracy Prediction
        final_prediction = np.argmax(predictions, axis = 1)
        true_labels = np.argmax(Y_batch, axis=1)
        boolean_predictions = final_prediction == true_labels
        batch_accuracy = np.sum(boolean_predictions) / batch_size
        

        # Calculate loss
        CCE_loss = Categorical_Cross_Entropy(predictions, Y_batch)
        loss_val = float(CCE_loss.get())

        #Backpropagation
        #Cross entropy loss and softmax derivative in one function
        grad_softmax = backpropagation_softmax(Y_batch, softmax_cache)

        #Output layer error
        grad_dense3_out, grad_W3, grad_b3 = linear_backward(grad_softmax, dense3_cache)

        #Second hidden layer error
        grad_dense2_out, grad_W2, grad_b2 = backpropagation_relu(grad_dense3_out, dense2_cache)

        #Hidden layer error
        new_grad_dense1 = (grad_dense2_out * mask_dense1) / 0.8

        grad_dense1_out, grad_W1, grad_b1 = backpropagation_relu(new_grad_dense1, dense1_cache)

        #Unflatten first hidden layer error
        grad_unflatten = backpropagation_unflatten(grad_dense1_out, flatten_cache)

        #Max Pool error
        grad_max_pool = max_pool_backpropagation(grad_unflatten, max_pool_cache)
        grad_relu_conv = relu_conv_backward(grad_max_pool, conv_relu_cache)

        #Convolutional layer error
        grad_F1, grad_b_conv = backpropagation_vectorized(grad_relu_conv, conv_cache)

        t += 1
        adam_memory = [lr, b1_momentum, b2_scaling, e, t]
        gradients = [grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, grad_F1, grad_b_conv]
        parameters = [W1, W2, W3, b1, b2, b3, F1, b_conv]
        # Send to optimizer to apply gradient descent
        parameters, adam_m, adam_u = Adam(adam_memory, gradients, parameters, adam_m, adam_u)

        W1, W2, W3, b1, b2, b3, F1, b_conv = parameters

        epoch_loss += loss_val
        epoch_accuracy += batch_accuracy.get() * 100
        batch_in_epoch += 1


    if epoch % 1 == 0:
        avg_loss = epoch_loss/batch_in_epoch
        avg_accuracy = epoch_accuracy/batch_in_epoch 
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")
    # Reduce lr every 10 epochs to find the highest accuracy
    if epoch % 10 == 0:
        lr = lr * 0.5

print("Attemp saving")
np.savez("trained_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, F1=F1, b_conv=b_conv)
print("Saved successfully")

print(f"Training finished. Final results: Loss: {CCE_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")



