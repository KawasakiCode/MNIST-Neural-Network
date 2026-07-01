from data import load_and_prep_data, augment_data
from network import MNIST
import tensorflow as tf

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

# Convert the numpy array's into tensorflow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.float32)

# Dataset.slices pairs the images from X_train with labels from Y_train
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, Y_train_tensor))
# .shuffle Shuffles the images and .batch creates the batches for the training
train_loader = train_dataset.shuffle(buffer_size=60000).batch(64)

# Initialize the model and run a fake forward pass
model = MNIST()
dummy = tf.zeros([1, 1, 28, 28])
model(dummy)

# Loss function (criterion)
# from_logits ensures that softmax also runs
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Put the train loop in a separate function to take advantage of tf.function's
# increased execution speed
@tf.function
def batch_loop(X_batch, Y_batch):
        # Gradient Tape watches all calculations to later compute the gradients
        with tf.GradientTape() as tape:
            
            # Forward Pass
            logits = model(X_batch, training=True)

            loss_value = loss_fn(Y_batch, logits)
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        
        predicted_classes = tf.argmax(logits, axis=1)
        true_classes = tf.argmax(Y_batch, axis=1)

        correct_predictions = tf.equal(predicted_classes, true_classes)
        correct_in_batch = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))

        dynamic_batch_size = tf.shape(X_batch)[0]

        return loss_value, correct_in_batch, dynamic_batch_size

# Per-epoch metric history for plotting (see plots/plot_curves.py)
history = {"epoch": [], "loss": [], "accuracy": []}

# Training loop
for epoch in range(30):
    running_loss = 0.0
    running_correct_predictions = 0
    total_samples = 0

    for X_batch, Y_batch in train_loader:
        # Augment per batch to ensure randomness
        X_batch_augmented = tf.py_function(func=augment_data, inp=[X_batch], Tout=tf.float32)
        X_batch_augmented.set_shape(X_batch.shape)

        loss_value, correct_in_batch, batch_size = batch_loop(X_batch_augmented, Y_batch)

        running_loss += loss_value.numpy()
        running_correct_predictions += correct_in_batch.numpy()
        total_samples += batch_size.numpy()

    # Record every epoch so the plotted curve is smooth...
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_correct_predictions / total_samples * 100
    history["epoch"].append(epoch + 1)
    history["loss"].append(float(epoch_loss))
    history["accuracy"].append(float(epoch_accuracy))
    # ...but only print every 5 to keep the console tidy.
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%")

model.save_weights("trained_model.weights.h5")

# Save metric history so the plotting scripts can draw loss/accuracy curves.
import json, os
os.makedirs("../metrics", exist_ok=True)
with open("../metrics/history_tensorflow.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved metric history to metrics/history_tensorflow.json")