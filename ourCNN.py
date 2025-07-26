import torch
from torch.utils.data import DataLoader
from torch import nn
import torchmetrics
from torchvision.transforms import transforms
import torchvision

# ----------------------------------
# Image preprocessing (data augmentation)
# ----------------------------------
transformer = transforms.Compose([
    transforms.Resize((150, 150)),                        # Resize images to 150x150
    transforms.RandomHorizontalFlip(),                    # Random horizontal flip
    transforms.RandomRotation(10),                        # Random small rotation
    transforms.ColorJitter(brightness=0.1,                # Random brightness change
                           contrast=0.1, 
                           saturation=0.1, 
                           hue=0.1),
    transforms.ToTensor(),                                # Convert image to tensor
    transforms.Normalize([0.5, 0.5, 0.5],                  # Normalize to [-1, 1]
                         [0.5, 0.5, 0.5])
])

# ----------------------------------
# Device configuration (GPU if available)
# ----------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# CNN Model Definition
# ----------------------------------
class OurCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(OurCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)  # Downsampling by 2

        # Additional convolutional layers
        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Fully connected layer
        self.fc = nn.Linear(32 * 75 * 75, num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu(output)

        output = torch.flatten(output, 1)  # Flatten before FC layer
        output = self.fc(output)

        return output

# ----------------------------------
# Initialize model and move to device
# ----------------------------------
model = OurCNN().to(device)

# ----------------------------------
# Training configuration
# ----------------------------------
epochs = 20
learning_rate = 0.001
train_path = '/lastDS/train/'   # Path to training dataset
test_path = '/lastDS/test/'     # Path to test dataset

# Load datasets using ImageFolder
train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transformer)
test_dataloader = DataLoader(test_dataset, batch_size=48, shuffle=True)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

# Accuracy metric using torchmetrics
metric = torchmetrics.Accuracy(task="multiclass", num_classes=12).to(device)

# ----------------------------------
# Training loop for one epoch
# ----------------------------------
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()               # Reset gradients
        pred = model(X)                     # Forward pass
        loss = loss_fn(pred, y)             # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights

        acc = metric(pred, y)               # Update accuracy metric

    acc = metric.compute()                  # Final training accuracy for the epoch
    print(f'Training accuracy at the end of the epoch: {acc}')
    metric.reset()                          # Reset metric state for next epoch

# ----------------------------------
# Evaluation loop
# ----------------------------------
def test_loop(dataloader, model, loss_fn):
    model.eval()
    with torch.no_grad():                  # No gradient needed for evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            acc = metric(pred, y)

    acc = metric.compute()                 # Final test accuracy
    print(f'Testing accuracy at the end of the epoch: {acc}')
    metric.reset()
    return acc

# ----------------------------------
# Training process with model saving
# ----------------------------------
best_accuracy = 0.0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_accuracy = test_loop(test_dataloader, model, loss_fn)

    # Save the model if current accuracy is the best
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), '/content/drive/MyDrive/cpu1.model')
        best_accuracy = test_accuracy

    print(f"Epoch n° {epoch} - Best acc: {best_accuracy.item()}")
