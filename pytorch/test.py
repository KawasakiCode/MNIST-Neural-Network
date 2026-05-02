import torch
from torch.utils.data import TensorDataset, DataLoader
from data import load_and_prep_data
from network import MNIST

TEST_FILEPATH = "mnist_test/mnist_test.csv"

X_test, Y_test = load_and_prep_data(TEST_FILEPATH)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = MNIST().to(device)

model.load_state_dict(torch.load("trained_model.pth", weights_only=True))

model.eval()

running_correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for X_batch, Y_batch in test_loader:

        X_batch_np = X_batch.cpu().numpy()

        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)

        Y_batch_indices = torch.argmax(Y_batch, dim=1)

        predictions = torch.argmax(logits, dim=1)
        running_correct_predictions += (predictions == Y_batch_indices).sum().item()
        total_samples += X_batch_tensor.size(0)

epoch_accuracy = (running_correct_predictions / total_samples) * 100
print(f"Testing complete. Accuracy: {epoch_accuracy:.2f}%")

