import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases, loss_derivatives, output_layer_derivatives, hidden_layer_derivatives, second_hidden_layer_derivatives
from activations import ReLU, Softmax
from losses import Categorical_Cross_Entropy
from optimizers import gradient_descent
from metrics import plot_training_curves, show_prediction

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

Y_raw = np.argmax(Y_train, axis = 1)

W1, W2, W3, b1, b2, b3 = initialize_weights_biases(512)

# Data to monitor training
loss_history = []
accuracy_history = []

print("Training loop started")
for epoch in range(1000):
    # First hidden layer
    Z1 = X_train @ W1 + b1
    A1 = ReLU(Z1)

    # Second hidden layer
    Z2 = A1 @ W2 + b2
    A2 = ReLU(Z2)

    #Output layer
    Z3 = A2 @ W3 + b3
    A3 = Softmax(Z3)

    # Accuracy Prediction
    final_prediction = np.argmax(A3, axis = 1)
    boolean_predictions = final_prediction == Y_raw
    accuracy = np.mean(boolean_predictions) * 100
    accuracy_history.append(accuracy)

    # Calculate loss
    CCE_loss = Categorical_Cross_Entropy(A3, Y_train)
    loss_history.append(CCE_loss)

    # Calculate derivatives with Chain Rule
    dZ3 = loss_derivatives(A3, Y_train)
    dW3, db3 = output_layer_derivatives(A2, dZ3)
    dZ2, dW2, db2 = second_hidden_layer_derivatives(dZ3, Z2, A1, W3)
    dW1, db1 = hidden_layer_derivatives(dZ2, W2, Z1, X_train)

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



