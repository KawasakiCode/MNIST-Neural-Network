import numpy as np
from data import load_and_prep_data
from network import initialize_weights_biases
from activations import ReLU, Softmax
from metrics import plot_accuracy_only, show_prediction

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

Y_raw = np.argmax(Y_test, axis = 1)

saved_data = np.load("trained_weights.npz")

W1 = saved_data['W1']
W2 = saved_data['W2']
b1 = saved_data['b1']
b2 = saved_data['b2']

# Data to monitor training
accuracy_history = []

print("Training loop started")
for epoch in range(1000):
    # The intermediate step of the hidden layer which is passed through ReLU
    Z1 = X_test @ W1 + b1

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

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Accuracy: {accuracy}%")


print(f"Training finished. Final results: Accuracy: {accuracy}%")
plot_accuracy_only(accuracy_history)

test_image = X_test[10]
test_pure_label = Y_raw[10]
test_prediction = final_prediction[10]

show_prediction(test_image, test_pure_label, test_prediction)