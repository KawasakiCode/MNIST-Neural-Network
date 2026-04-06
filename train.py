import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases, loss_derivatives, output_layer_derivatives, hidden_layer_derivatives
from activations import ReLU, Softmax
from losses import Categorical_Cross_Entropy
from optimizers import gradient_descent
from metrics import plot_training_curves, show_prediction

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

Y_raw = np.argmax(Y_train, axis = 1)

W1, W2, b1, b2 = initialize_weights_biases(128)

# Data to monitor training
loss_history = []
accuracy_history = []

print("Training loop started")
for epoch in range(1000):
    # The intermediate step of the hidden layer which is passed through ReLU
    Z1 = X_train @ W1 + b1

    # The output of the hidden layer
    A1 = ReLU(Z1)

    # Pass the output of the hidden layer into the output layer
    Z2 = A1 @ W2 + b2

    # Final output of the network
    A2 = Softmax(Z2)
    final_prediction = np.argmax(A2, axis = 1)
    boolean_predictions = final_prediction == Y_raw
    accuracy = np.mean(boolean_predictions) * 100
    accuracy_history.append(accuracy)

    # Calculate loss
    CCE_loss = Categorical_Cross_Entropy(A2, Y_train)
    loss_history.append(CCE_loss)

    # Calculate derivatives with Chain Rule
    dZ2 = loss_derivatives(A2, Y_train)
    dW2, db2 = output_layer_derivatives(A1, dZ2)
    dW1, db1 = hidden_layer_derivatives(dZ2, W2, Z1, X_train)

    # Send to optimizer to apply gradient descent
    lr = 0.01
    W1, W2, b1, b2 = gradient_descent(dW1, dW2, db1, db2, W1, W2, b1, b2, lr)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {CCE_loss}, Accuracy: {accuracy}%")


print(f"Training finished. Final results: Loss: {CCE_loss}, Accuracy: {accuracy}%")
plot_training_curves(loss_history, accuracy_history)

test_image = X_train[10]
test_pure_label = Y_raw[10]
test_prediction = final_prediction[10]

show_prediction(test_image, test_pure_label, test_prediction)

np.savez("trained_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)

