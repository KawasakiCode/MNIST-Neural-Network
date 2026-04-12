import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases, initialize_cnn_filters
from forward import conv_forward, linear_forward, relu_forward, flatten_forward, softmax_forward
from backpropagation import backpropagation_conv, backpropagation_relu, backpropagation_softmax, backpropagation_unflatten, linear_backward
from losses import Categorical_Cross_Entropy
from optimizers import gradient_descent
from metrics import plot_training_curves

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

Y_raw = np.argmax(Y_train, axis = 1)

W1, W2, W3, b1, b2, b3 = initialize_weights_biases(512)
F1, b_conv = initialize_cnn_filters(8)

# Data to monitor training
loss_history = []
accuracy_history = []

print("Training loop started")
for epoch in range(1000):
    # Pass the data through the convolutional layer
    conv_output_data, conv_cache = conv_forward(X_train, F1, b_conv)
    flattened_data, flatten_cache = flatten_forward(conv_output_data)

    # Pass through first hidden layer
    dense1_output, dense1_cache = relu_forward(flattened_data, W1, b1)

    # Pass through second hidden layer
    dense2_output, dense2_cache = relu_forward(dense1_output, W2, b2)

    # Output layer (Predictions)
    dense3_output, dense3_cache = linear_forward(dense2_output, W3, b3)
    predictions, softmax_cache = softmax_forward(dense3_output)

    # Accuracy Prediction
    final_prediction = np.argmax(predictions, axis = 1)
    boolean_predictions = final_prediction == Y_raw
    accuracy = np.mean(boolean_predictions) * 100
    accuracy_history.append(accuracy)

    # Calculate loss
    CCE_loss = Categorical_Cross_Entropy(predictions, Y_train)
    loss_history.append(CCE_loss)

    #Backpropagation
    #Cross entropy loss and softmax derivative in one function
    grad_softmax = backpropagation_softmax(Y_train, softmax_cache)

    #Output layer error
    grad_dense3_out, grad_W3, grad_b3 = linear_backward(grad_softmax, dense3_cache)

    #Second hidden layer error
    grad_dense2_out, grad_W2, grad_b2 = backpropagation_relu(grad_dense3_out, dense2_cache)

    #Hidden layer error
    grad_dense1_out, grad_W1, grad_b1 = backpropagation_relu(grad_dense2_out, dense1_cache)

    #Unflatten first hidden layer error
    grad_unflatten = backpropagation_unflatten(grad_dense1_out, flatten_cache)

    #Convolutional layer error
    grad_F1, grad_b_conv = backpropagation_conv(grad_unflatten, conv_cache)

    # Send to optimizer to apply gradient descent
    lr = 0.01
    W1, W2, W3, b1, b2, b3, F1, b_conv = gradient_descent(grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, W1, W2, W3, b1, b2, b3, F1, b_conv, grad_F1, grad_b_conv, lr)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {CCE_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Attemp saving")
np.savez("trained_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("Saved successfully")

print(f"Training finished. Final results: Loss: {CCE_loss:.4f}, Accuracy: {accuracy:.2f}%")
plot_training_curves(loss_history, accuracy_history)



