import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases, initialize_cnn_filters
from forward import conv_forward, linear_forward, relu_forward, flatten_forward, softmax_forward
from backpropagation import backpropagation, backpropagation_softmax
from activations import ReLU, Softmax
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

    # Calculate derivatives with Chain Rule
    #TODO make them with the new functions
    # dZ3 = loss_derivatives(A3, Y_train)
    # dW3, db3 = output_layer_derivatives(A2, dZ3)
    # dZ2, dW2, db2 = second_hidden_layer_derivatives(dZ3, Z2, A1, W3)
    # dW1, db1 = hidden_layer_derivatives(dZ2, W2, Z1, X_train)

    # Send to optimizer to apply gradient descent
    lr = 0.01
    W1, W2, W3, b1, b2, b3 = gradient_descent(dW1, dW2, dW3, db1, db2, db3, W1, W2, W3, b1, b2, b3, lr)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {CCE_loss:.4f}, Accuracy: {accuracy:.2f}%")

print("Attemp saving")
np.savez("trained_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
print("Saved successfully")

print(f"Training finished. Final results: Loss: {CCE_loss:.4f}, Accuracy: {accuracy:.2f}%")
plot_training_curves(loss_history, accuracy_history)



