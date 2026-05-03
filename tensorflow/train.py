from data import load_and_prep_data, augment_data
from network import MNIST
import tensorflow as tf

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

# Convert the numpy array's into tensorflow tensors
X_train_tensor = tf.convert_to_tensor(X_train.get(), dtype=tf.float32)
Y_train_tensor = tf.convert_to_tensor(Y_train.get(), dtype=tf.float32)

# Dataset.slices pairs the images from X_train with labels from Y_train
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, Y_train_tensor))
# .shuffle Shuffles the images and .batch creates the batches for the training
train_loader = train_dataset.shuffle(buffer_size=60000).batch(64)

# Initialize the model
model = MNIST()

# Loss function (criterion)
# from_logits ensures that softmax also runs
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(30):
    running_loss = 0.0
    running_correct_predictions = 0
    total_samples = 0

    for X_batch, Y_batch in train_loader:
        
        # Augment per batch to ensure randomness
        X_batch_augmented = augment_data(X_batch)

        # Gradient Tape watches all calculations to later compute the gradients
        with tf.GradientTape() as tape:
            
            # Forward Pass
            logits = model(X_batch_augmented, training=True)

            loss_value = loss_fn(Y_batch, logits)
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        running_loss += loss_value.numpy()
        predicted_classes = tf.argmax(logits, axis=1)
        true_classes = tf.argmax(Y_batch, axis=1)

        correct_predictions = tf.equal(predicted_classes, true_classes)
        correct_in_batch = tf.reduce_sum(tf.cast(correct_predictions, tf.int32))
        running_correct_predictions += correct_in_batch.numpy()
        total_samples += X_batch.shape[0]
    
    if epoch % 1 == 0:
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = running_correct_predictions / total_samples
        print(f"Epoch {epoch+1} Loss: {epoch_loss} Accuracy: {epoch_accuracy}")




