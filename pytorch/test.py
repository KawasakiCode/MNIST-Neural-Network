import torch
from torch.utils.data import TensorDataset, DataLoader
from data import load_and_prep_data
from network import MNIST

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

# Convert numpy arrays into pytorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# Pick cuda as the device if available and send the tensors there
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)

# TensorDataset combines the arrays so that X[0] is always paired with Y[0]
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
# DataLoader splits the data into batches of 64. Shuffle is false in test cause
# order doesn't matter in testing
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model and load weights
model = MNIST().to(device)
model.load_state_dict(torch.load("trained_model.pth", weights_only=True))
# Put model in eval mode (testing mode) so it doesnt keep track of gradients
model.eval()

running_correct_predictions = 0
total_samples = 0

# Test forward pass
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        # Send the tensors to device because TensorDataset creates them in cpu
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)

        # Labels and logits are in this format: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] so we crush them into [2]
        # to be easier to compare them
        Y_batch_indices = torch.argmax(Y_batch, dim=1)
        predictions = torch.argmax(logits, dim=1)
        running_correct_predictions += (predictions == Y_batch_indices).sum().item()

        total_samples += X_batch.size(0)

epoch_accuracy = (running_correct_predictions / total_samples) * 100
print(f"Testing complete. Accuracy: {epoch_accuracy:.2f}%")

