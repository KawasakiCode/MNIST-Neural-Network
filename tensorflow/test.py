from data import load_and_prep_data
import tensorflow as tf
from network import MNIST

tf.keras.backend.set_image_data_format('channels_first')

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

# Convert the numpy array's into tensorflow tensors
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test_tensor = tf.convert_to_tensor(Y_test, dtype=tf.float32)

# Dataset.slices pairs the images from X_train with labels from Y_train
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_tensor, Y_test_tensor))
# .shuffle Shuffles the images and .batch creates the batches for the training
test_loader = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

model = MNIST()
dummy = tf.zeros([1, 1, 28, 28])
model(dummy)
model.build((None, 1, 28, 28))
model.load_weights("trained_model.weights.h5")

# Loss function (criterion)
# from_logits ensures that softmax also runs
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

@tf.function
def test_loop(X_batch, Y_batch):

    logits = model(X_batch, training=False)

    loss_value = loss_fn(Y_batch, logits)

    predicted_classes = tf.argmax(logits, axis=1)
    true_classes = tf.argmax(Y_batch, axis=1)
    correct_predictions = tf.equal(predicted_classes, true_classes)
    correct_in_batch = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

    dynamic_batch_size = tf.shape(X_batch)[0]

    return loss_value, correct_in_batch, dynamic_batch_size

running_loss = 0.0
running_correct_predictions = 0
total_samples = 0

for X_batch, Y_batch in test_loader:
    
    loss_value, correct_in_batch, batch_size = test_loop(X_batch, Y_batch)

    running_loss += loss_value.numpy()
    running_correct_predictions += correct_in_batch.numpy()
    total_samples += batch_size.numpy()

test_loss = running_loss / len(test_loader)
epoch_accuracy = running_correct_predictions / total_samples * 100
print(f"Test Loss: {test_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")