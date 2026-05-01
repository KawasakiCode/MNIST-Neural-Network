import torch.nn as nn

class MNIST(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layer
        # 1 color channel
        # 8 filters
        # 3x3 filter size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)

        # ReLU
        self.relu = nn.ReLU()

        # Max Pool layer
        # 2x2 block size
        # Stride 2 (match block size)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense Layers
        # Flatten Max Pool output
        self.flatten = nn.Flatten()

        # Hidden Layer 1
        self.dense1 = nn.Linear(in_features=1352, out_features=128)

        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.2)

        # Hidden Layer 2
        self.dense2 = nn.Linear(in_features=128, out_features=10)
    
    def forward(self, x):
        # Forward Pass function
        # Pytorch automatically stores cache

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.dense2(x)

        return logits