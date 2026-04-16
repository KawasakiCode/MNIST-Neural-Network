import cupy as np

def gradient_descent(dW1: np.ndarray, dW2: np.ndarray, dW3: np.ndarray, db1: np.ndarray, db2: np.ndarray, db3: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, F1: np.ndarray, b_conv: np.ndarray, dF1: np.ndarray, db_conv: np.ndarray, learning_rate: float):
    # Calculate new weights
    W1_new = W1 - (learning_rate * dW1)
    W2_new = W2 - (learning_rate * dW2)
    W3_new = W3 - (learning_rate * dW3)

    # Calculate new biases
    b1_new = b1 - (learning_rate * db1)
    b2_new = b2 - (learning_rate * db2)
    b3_new = b3 - (learning_rate * db3)

    #Calculate filters weights, biases
    F1_new = F1 - (learning_rate * dF1)
    b_conv_new = b_conv - (learning_rate * db_conv)


    return W1_new, W2_new, W3_new, b1_new, b2_new, b3_new, F1_new, b_conv_new

def Adam(adam_memory, gradients, parameters):
    lr, b1_momentum, b2_scaling_decay, e, t = adam_memory
    dW1, dW2, dW3, db1, db2, db3, dF1, b_conv = gradients 
    W1, W2, W3, F1, b1, b2, b3, b_conv = parameters
