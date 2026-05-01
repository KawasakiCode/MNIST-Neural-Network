import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from data import augment_data, load_and_prep_data
from network import MNIST

TRAIN_FILEPATH = "mnist_train/mnist_train.csv"

X_train, Y_train = load_and_prep_data(TRAIN_FILEPATH)

X_augmented = augment_data(X_train)

X_augmented_tensor = torch.tensor(X_augmented, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train_tensor = X_augmented_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)

train_dataset = TensorDataset(X_augmented_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = MNIST().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(30):
    running_loss = 0.0
    running_correct_predictions = 0
    total_samples = 0

    for X_batch, Y_batch in train_loader:

        X_batch_np = X_batch.cpu().numpy()

        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        Y_batch_indices = torch.argmax(Y_batch, dim=1)

        predictions = torch.argmax(logits, dim=1)
        running_correct_predictions += (predictions == Y_batch_indices).sum().item()
        running_loss += loss.item() * X_batch_tensor.size(0)
        total_samples += X_batch_tensor.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        epoch_loss = running_loss / total_samples
        epoch_accuracy = (running_correct_predictions / total_samples) * 100
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.2f}, Accuracy: {epoch_accuracy:.2f}%")
    
torch.save(model.state_dict(), "trained_model.pth")