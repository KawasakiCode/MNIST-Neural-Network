import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data import augment_data, load_and_prep_data
from network import MNIST

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

# Convert numpy arrays into pytorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

# Pick cuda as the device if available and send the tensors there
device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)

# TensorDataset combines the arrays so that X[0] is always paired with Y[0]
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# DataLoader splits the data into batches of 64. Shuffle is true in training cause
# order does matter because the network may memorize the order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
model = MNIST().to(device)

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Per-epoch metric history for plotting (see plots/plot_curves.py)
history = {"epoch": [], "loss": [], "accuracy": []}

# Training loop
for epoch in range(30):
    running_loss = 0.0
    running_correct_predictions = 0
    total_samples = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        X_batch_augmented = augment_data(X_batch)

        logits = model(X_batch_augmented)
        loss = criterion(logits, Y_batch)

        # Labels and logits are in this format: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] so we crush them into [2]
        # to be easier to compare them
        Y_batch = Y_batch.to(device)
        Y_batch_indices = torch.argmax(Y_batch, dim=1)
        predictions = torch.argmax(logits, dim=1)
        running_correct_predictions += (predictions == Y_batch_indices).sum().item()

        running_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

        # Reset optimizer, calculate gradients with backpropagation and change them
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        epoch_loss = running_loss / total_samples
        epoch_accuracy = (running_correct_predictions / total_samples) * 100
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.2f}, Accuracy: {epoch_accuracy:.2f}%")
        history["epoch"].append(epoch + 1)
        history["loss"].append(float(epoch_loss))
        history["accuracy"].append(float(epoch_accuracy))

# Save model weights
torch.save(model.state_dict(), "trained_model.pth")

# Save metric history so the plotting scripts can draw loss/accuracy curves.
import json, os
os.makedirs("../metrics", exist_ok=True)
with open("../metrics/history_pytorch.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved metric history to metrics/history_pytorch.json")