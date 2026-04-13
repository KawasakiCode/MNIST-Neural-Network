import cupy as np

def ReLU(matrix: np.ndarray) -> np.ndarray:
    # np.maximum iterates through the whole array
    return np.maximum(0, matrix)

def Softmax(matrix: np.ndarray) -> np.ndarray:
    # Subtract the max from each row of the array to prevent 
    # softmax from doing e^x with very big x which would 
    # result in memory overflow

    max_matrix = np.max(matrix, axis = 1, keepdims = True)
    stable_matrix = matrix - max_matrix

    softmax_numerator = np.exp(stable_matrix)
    softmax_denominator = np.sum(softmax_numerator, axis = 1, keepdims = True)

    return softmax_numerator / softmax_denominator
