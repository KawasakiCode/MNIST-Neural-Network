import numpy as np

def Categorical_Cross_Entropy(output_matrix: np.ndarray, labels_matrix: np.ndarray):
    # We use .clip to ensure no number in the output matrix is 
    # equal to 0 to prevent log(0) from happening
    safe_output_matrix = np.clip(output_matrix, 1e-7, 1 - 1e-7)

    result_matrix = np.log(safe_output_matrix) * labels_matrix

    loss_sum = np.sum(result_matrix)

    cce = -1 * (loss_sum/output_matrix.shape[0])
    return cce

