import numpy as np
from data import load_and_prep_data
from activations import ReLU, Softmax

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

Y_raw = np.argmax(Y_test, axis = 1)

saved_data = np.load("trained_weights.npz")

W1 = saved_data['W1']
W2 = saved_data['W2']
W3 = saved_data['W3']
b1 = saved_data['b1']
b2 = saved_data['b2']
b3 = saved_data['b3']

# Data to monitor training
accuracy_history = []

# The intermediate step of the hidden layer which is passed through ReLU
Z1 = X_test @ W1 + b1
A1 = ReLU(Z1)

# Second hidden layer
Z2 = A1 @ W2 + b2
A2 = ReLU(Z2)

#Output layer
Z3 = A2 @ W3 + b3
A3 = Softmax(Z3)

final_prediction = np.argmax(A3, axis = 1)
boolean_predictions = final_prediction == Y_raw
accuracy = np.mean(boolean_predictions) * 100
accuracy_history.append(accuracy)

print(f"Testing finished. Final results: Accuracy: {accuracy}%")