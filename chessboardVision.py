import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
from chessboard import display

# Image preprocessing pipeline
transformer = transforms.Compose([
    transforms.Resize((150, 150)),                          # Resize image to 150x150
    transforms.RandomHorizontalFlip(),                      # Add randomness by flipping horizontally
    transforms.RandomRotation(10),                          # Add robustness with slight rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                           saturation=0.1, hue=0.1),        # Random color distortion
    transforms.ToTensor(),                                  # Convert to PyTorch tensor
    transforms.Normalize([0.5, 0.5, 0.5], 
                         [0.5, 0.5, 0.5])                   # Normalize to [-1, 1]
])

# Set device for inference (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the CNN architecture
class OurCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(OurCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

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

        output = torch.flatten(output, 1)  # Flatten for fully connected layer
        output = self.fc(output)

        return output

# Load the pre-trained model
model = OurCNN().to(device)
model.load_state_dict(torch.load('gpu2.model', map_location="cpu"))
model.eval()  # Set model to evaluation mode

# Function to detect and warp the chessboard from the image
def warpBoard(img_copy):
    src_points = []

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to highlight chessboard
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 7)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area > 20000:  # Large area likely to be the board
            filtered_contours.append(contour)

    canvas = img_copy.copy()

    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        src_points.append((x, y))
        src_points.append((x, h + y))
        src_points.append((w + x, h + y))
        src_points.append((w + x, y))

    # Show rectangle drawn around detected chessboard
    cv2.imshow("Chessboard Boxes", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Destination points for warping
    dst_points = np.array([(0, 0), (0, 600), (600, 600), (600, 0)], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
    
    return perspective_matrix

# Function to detect squares and assign coordinates to chess notation
def setSquares(img_warped):
    gray_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Highlight squares
    gray_warped = cv2.bitwise_and(img_warped, img_warped, mask=closing)

    edges = cv2.Canny(gray_warped, 120, 180)
    thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 7, 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
        if len(approx) == 4:
            squares.append(approx.reshape(-1, 2))

    sq_coords = []
    for square in squares:
        area = cv2.contourArea(square)
        if 2500 < area < 5000:
            cv2.polylines(img_warped, [square], True, (0, 255, 0), 2)
            coordinates = [tuple(point) for point in square.tolist()]
            sorted_coords = sorted(coordinates, key=lambda point: (point[0] + point[1], point[0] - point[1]))
            sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]
            sq_coords.append(sorted_coords)

    print(len(sq_coords))

    # Create dictionary for chessboard squares
    board = {f"{file}{rank}": None for rank in "87654321" for file in "abcdefgh"}
    keys = list(board.keys())

    # Sort square coordinates row-wise and column-wise
    def sort_coordinates(coords):
        sorted_coords = sorted(coords, key=lambda point: (point[0] + point[1], point[0] - point[1]))
        sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]
        return sorted_coords

    sorted_sq_coords = sorted(sq_coords, key=lambda coords: (coords[0][1], coords[0][0]))
    rows = [sorted_sq_coords[i:i+8] for i in range(0, len(sorted_sq_coords), 8)]

    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda coords: coords[0][0])

    sorted_sq_coords = [coords for row in rows for coords in row]

    # Map square coordinates to board dictionary
    for i, key in enumerate(keys):
        if i < len(sorted_sq_coords):
            board[key] = sort_coordinates(sorted_sq_coords[i])

    return img_warped, board

# Main function to run live detection and analysis
def main():
    game_board = display.start()
    starting_fen = '8/8/8/8/8/8/8/8 w - - 0 1'
    display.update(starting_fen, game_board)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    end_loop = False
    perspective_matrix = None
    k = None
    board = {}

    while not end_loop:
        display.check_for_quit()
        _, img = cap.read()
        if img is None:
            continue

        img_copy = cv2.resize(img.copy(), (600, 600))

        if perspective_matrix is None and k == 13:  # ENTER pressed
            perspective_matrix = warpBoard(img_copy)
            img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))
            mask, board = setSquares(img_warped)
            cv2.imshow("ChessBoard", img_warped)
            cv2.imshow("Boxes", mask)

        elif perspective_matrix is not None and k == 32:  # SPACE pressed
            fenN = ""
            squares_to_check = [
                *[f"{f}8" for f in "abcdefgh"],
                *[f"{f}7" for f in "abcdefgh"],
                *[f"{f}2" for f in "abcdefgh"],
                *[f"{f}1" for f in "abcdefgh"]
            ]

            for square in squares_to_check:
                img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))
                cv2_to_pil = cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(cv2_to_pil)
                sq = img_pil.crop((board[square][0][0], board[square][0][1],
                                   board[square][2][0], board[square][2][1]))

                square_tensor = transformer(sq).unsqueeze(0)

                with torch.no_grad():
                    output = model(square_tensor)

                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                chess_pieces = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
                predicted_piece = chess_pieces[predicted_class]

                fenN += predicted_piece
                if square.endswith("h8") or square.endswith("h7") or square.endswith("h2"):
                    fenN += "/"
                    if square.endswith("7"):
                        fenN += "8/8/8/8/"

            fenN += " w KQkq - 0 1"
            display.update(fenN, game_board)

        elif perspective_matrix is not None and k == 8:  # BACKSPACE pressed
            # Placeholder for move detection and board update
            pass

        elif perspective_matrix is not None:
            img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))

            # Mouse callback to print clicked square
            def cheCasella(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    for square_name, square_coords in board.items():
                        if square_coords[0] is None:
                            continue
                        top_left = square_coords[0]
                        bottom_right = square_coords[2]
                        if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                            print(square_name)
                            return

            cv2.setMouseCallback("ChessBoard", cheCasella)
            cv2.imshow("ChessBoard", img_warped)

        else:
            cv2.imshow("ChessBoard", img_copy)

        k = cv2.waitKey(20)
        if k == 27:  # ESC key
            end_loop = True

    cap.release()
    cv2.destroyAllWindows()
    display.terminate()

# Entry point
if __name__ == "__main__":
    main()
