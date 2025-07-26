import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ConvNet, self).__init__()
        
        # Formula for output size after convolution:
        # ((w - f + 2P) / s) + 1
        
        # Input shape: (batch_size=256, channels=3, height=150, width=150)
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Batch normalization keeps shape: (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        # Activation: (256, 12, 150, 150)
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Max pooling reduces height and width by factor 2
        # Output shape: (256, 12, 75, 75)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 20, 75, 75)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        
        # Flatten the tensor to feed into fully connected layer
        # Shape: (batch_size, 32 * 75 * 75)
        output = output.view(-1, 32 * 75 * 75)
        
        output = self.fc(output)
        
        return output

# Define image transformations: resize, tensor conversion, normalization
transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    
    # Optional augmentations for training (commented out):
    # transforms.RandomResizedCrop((150, 150), scale=(0.8, 1.0)),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomRotation(15),
    # transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the trained model weights
model = ConvNet(num_classes=12)
model.load_state_dict(torch.load('lastmodel2.model', map_location="cpu"))
model.eval()  # Set model to evaluation mode

# Load and preprocess the input image
image_path = 'C:/Users/User/Desktop/ai lab/Progetto/myds/lastDS/test/bk/5.png'  # Change path as needed
image = Image.open(image_path).convert("RGB")
image = transformer(image).unsqueeze(0)  # Add batch dimension: (1, 3, 150, 150)

# Perform inference without calculating gradients
with torch.no_grad():
    output = model(image)

# Get predicted class index with highest score
score, predicted = torch.max(output, 1)
print(output)  # Model raw outputs (logits)

predicted_class = predicted.item()

# Map predicted index to chess piece label
chess_pieces = ["bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"]
predicted_piece = chess_pieces[predicted_class]

print("Predicted chess piece:", predicted_piece)
