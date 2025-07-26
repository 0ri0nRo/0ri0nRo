import cv2
import numpy as np
from os import mkdir

# ----------------------------------------
# Detects the chessboard and applies perspective transformation
# ----------------------------------------
def warpBoard(img_copy):
    src_points = []

    # Convert image to grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to emphasize contours
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 7
    )

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to isolate the chessboard
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 20000:
            filtered_contours.append(contour)

    # Draw rectangles and store corners for transformation
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Manually define 4 corners: TL, BL, BR, TR
        src_points.append((x, y))
        src_points.append((x, y + h))
        src_points.append((x + w, y + h))
        src_points.append((x + w, y))

    # Show preview with detected board (debugging)
    cv2.imshow("Chessboard Boxes", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Define destination square (600x600)
    dst_points = np.array([(0, 0), (0, 600), (600, 600), (600, 0)], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)

    return perspective_matrix

# ----------------------------------------
# Detect and sort squares on the warped chessboard
# ----------------------------------------
def setSquares(img_warped):
    gray_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate white squares
    _, binary = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Mask the gray squares
    gray_warped = cv2.bitwise_and(img_warped, img_warped, mask=closing)

    # Detect edges
    edges = cv2.Canny(gray_warped, 120, 180)

    # Refine binary image again
    thresh = cv2.adaptiveThreshold(
        edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 9, 1
    )

    # Find contours of all potential squares
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
            coords = [tuple(point) for point in square.tolist()]
            sorted_coords = sorted(coords, key=lambda p: (p[0] + p[1], p[0] - p[1]))
            sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]
            sq_coords.append(sorted_coords)

    # Initialize an empty board dictionary
    board = {
        f"{file}{rank}": None
        for rank in range(8, 0, -1)
        for file in "abcdefgh"
    }
    keys = list(board.keys())

    def sort_coordinates(coords):
        sorted_coords = sorted(coords, key=lambda p: (p[0] + p[1], p[0] - p[1]))
        sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]
        return sorted_coords

    # Sort coordinates top to bottom, left to right
    sorted_sq_coords = sorted(sq_coords, key=lambda c: (c[0][1], c[0][0]))
    rows = [sorted_sq_coords[i:i+8] for i in range(0, len(sorted_sq_coords), 8)]
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda c: c[0][0])
    sorted_sq_coords = [coord for row in rows for coord in row]

    # Assign coordinates to board positions
    for i, key in enumerate(keys):
        if i < len(sorted_sq_coords):
            board[key] = sort_coordinates(sorted_sq_coords[i])

    return img_warped, board

# ----------------------------------------
# Save cropped square images to disk
# ----------------------------------------
def salvaFoto(board, img_warped):
    for square_name, coords in board.items():
        x1, y1 = coords[0]
        x2, y2 = coords[2]
        square = img_warped[y1:y2, x1:x2]
        square = cv2.resize(square, (150, 150))

        # Create directory and save original + rotated versions
        mkdir(square_name)
        cv2.imwrite(f"{square_name}/1.png", square)
        for i in range(2, 5):
            square = cv2.rotate(square, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(f"{square_name}/{i}.png", square)

# ----------------------------------------
# Main entry point
# ----------------------------------------
def main():
    img = cv2.imread("miav.jpg")
    img_copy = cv2.resize(img.copy(), (600, 600))

    perspective_matrix = warpBoard(img_copy)
    img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))
    mask, board = setSquares(img_warped)

    # Uncomment to save square images
    # salvaFoto(board, img_warped)

if __name__ == "__main__":
    main()
