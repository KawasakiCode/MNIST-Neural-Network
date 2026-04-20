import cupy as np

def SGD(gradients, parameters, learning_rate):
    # Calculate new weights
    for i in range(len(parameters)):
        parameters[i] = parameters[i] - (learning_rate * gradients[i])

    return parameters, gradients

def Adam(adam_memory, gradients, parameters, adam_m, adam_u):
    lr, b1_momentum, b2_scaling, e, t = adam_memory

    for i in range(len(parameters)):
        # Update m (Velocity)
        adam_m[i] = b1_momentum *  adam_m[i] + (1 - b1_momentum) * gradients[i]
        # Update u (Friction)
        adam_u[i] = b2_scaling * adam_u[i] + (1 - b2_scaling) * (gradients[i] ** 2)
        # Apply bias correction
        m_hat = adam_m[i] / (1 - b1_momentum ** t)
        u_hat = adam_u[i] / (1 - b2_scaling ** t)
        # Final Parameter Update
        parameters[i] = parameters[i] - lr * (m_hat / (np.sqrt(u_hat) + e))

    return parameters, adam_m, adam_u



