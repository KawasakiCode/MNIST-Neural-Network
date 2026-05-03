import cupy as np
import tensorflow as tf

def load_and_prep_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)

    # First column (labels)
    Y_raw = data[:, 0]

    # Pixel data (28x28 = 784 values)
    X_raw = data[:, 1:]

    # Normalize values so they are between 0.0 and 1.0
    # Neural networks work faster with smaller numbers
    X = X_raw / 255.0

    m = X.shape[0]

    X_train = X.reshape(m, 1, 28, 28)

    # Create an array of zeros (number of images x 10)
    num_images = Y_raw.shape[0]
    Y_encoded = np.zeros((num_images, 10))

    Y_encoded[np.arange(num_images), Y_raw.astype(int)] = 1.0

    return X_train, Y_encoded

def augment_data(data):
    x_np = data.numpy()
    shift_y = np.random.randint(-2, 3)
    shift_x = np.random.randint(-2, 3)

    shifted_data = np.zeros_like(x_np)

    dest_y1 = max(0, shift_y)
    dest_y2 = min(28, 28 + shift_y)
    dest_x1 = max(0, shift_x)
    dest_x2 = min(28, 28 + shift_x)

    src_y1 = max(0, -shift_y)
    src_y2 = min(28, 28 - shift_y)
    src_x1 = max(0, -shift_x)
    src_x2 = min(28, 28 - shift_x)

    shifted_data[:, :, dest_y1:dest_y2, dest_x1:dest_x2] = x_np[:, :, src_y1:src_y2, src_x1:src_x2]
    shifted_data_tf = tf.convert_to_tensor(shifted_data, dtype=tf.float32)

    return shifted_data_tf