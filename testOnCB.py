import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import cv2

class ConvNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ConvNet, self).__init__()
        
        # Formula for output size after convolution:
        # ((w - f + 2P) / s) + 1

        # Input shape example: (batch=256, channels=3, height=150, width=150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 12, 150, 150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Batch norm keeps shape: (256, 12, 150, 150)
        self.relu1 = nn.ReLU()
        # Activation output: (256, 12, 150, 150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Max pooling reduces spatial size by half
        # Output shape: (256, 12, 75, 75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 20, 75, 75)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output shape: (256, 32, 75, 75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        # Fully connected layer to classify into num_classes categories
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

        # Flatten output tensor for the fully connected layer
        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output

# Image transformations including augmentation and normalization
transformer = transforms.Compose([
    transforms.Resize((150, 150)),

    # Augmentations for training (can be commented out for testing)
    transforms.RandomResizedCrop((150, 150), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),

    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the trained model weights
model = ConvNet(num_classes=12)
model.load_state_dict(torch.load('colabmodel4.model', map_location="cpu"))
model.eval()  # Set model to evaluation mode

# Load the full chessboard image
image_path = 'C:/Users/User/Desktop/ai lab/Progetto/chess.png'  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Define the size of each square and the step between squares (stride)
square_size = 140  # Width and height of each square region to crop
stride = 140       # Move by this amount to get next square (no overlap here)

# Get image dimensions
width, height = image.size

# List to keep predictions for each detected square
predictions = []

# Slide over the image with windows of size square_size
for y in range(20, height - square_size + 1, stride):
    for x in range(20, width - square_size + 1, stride):
        # Crop a square from the big image
        square = image.crop((x, y, x + square_size, y + square_size))
        
        # Preprocess the cropped square image with the transformer
        square_tensor = transformer(square).unsqueeze(0)  # Add batch dimension
        
        # Predict the chess piece on this square
        with torch.no_grad():
            output = model(square_tensor)
        
        # Get the highest score class
        score, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        
        # Map the predicted class index to a chess piece label
        chess_pieces = ["bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"]
        predicted_piece = chess_pieces[predicted_class]
        
        # Store prediction with the bounding box coordinates
        predictions.append((x, y, x + square_size, y + square_size, predicted_piece))

# Load image in OpenCV to draw rectangles and labels
image_cv = cv2.imread(image_path)
color = (0, 255, 0)  # Green color in BGR

# Draw rectangles and labels on the image for each predicted square
for (x1, y1, x2, y2, piece) in predictions:
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, thickness=2)
    
    label = f"{piece}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
    label_x = x1 + (square_size - label_size[0]) // 2
    label_y = y2 - label_size[1] - 5
    cv2.putText(image_cv, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)

# Show the image with bounding boxes and piece labels
cv2.namedWindow("Detected Pieces", cv2.WINDOW_KEEPRATIO)
cv2.imshow("Detected Pieces", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
