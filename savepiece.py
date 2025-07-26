import cv2
import numpy as np
from os import mkdir

# ----------------------------------------
# Compute the perspective transform matrix for the chessboard
# ----------------------------------------
def warpBoard(img_copy):
    src_points = []

    # Convert image to grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary (inverted) image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 7
    )

    # Find external contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area (select the largest contour, which should be the chessboard)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 20000:
            filtered_contours.append(contour)

    # Create a copy of the image to draw rectangles on
    canvas = img_copy.copy()

    # Draw rectangles around filtered contours and save their corner points
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

        src_points.append((x, y))        # Top-left
        src_points.append((x, y + h))    # Bottom-left
        src_points.append((x + w, y + h))# Bottom-right
        src_points.append((x + w, y))    # Top-right

    # Show the image with detected chessboard rectangle (for debugging)
    cv2.imshow("Chessboard Boxes", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Destination points for perspective transform (square 600x600)
    dst_points = np.array([(0, 0), (0, 600), (600, 600), (600, 0)], dtype=np.float32)

    # Calculate perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)

    return perspective_matrix


# ----------------------------------------
# Detect and sort the squares on the warped chessboard image
# ----------------------------------------
def setSquares(img_warped):

    # Convert warped image to grayscale
    gray_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    # Threshold with Otsu to get binary inverted image
    _, binary = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological opening and closing to clean noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Mask the warped image to emphasize darker squares
    gray_warped = cv2.bitwise_and(img_warped, img_warped, mask=closing)

    # Detect edges
    edges = cv2.Canny(gray_warped, 120, 180)

    # Adaptive threshold on edges to isolate contours
    thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 1)

    # Find contours in thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for contour in contours:
        # Approximate contour with polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
        # Check if polygon has 4 sides (potential square)
        if len(approx) == 4:
            squares.append(approx.reshape(-1, 2))

    sq_coords = []

    # Filter squares by area and sort their corners (TL, BL, BR, TR)
    for square in squares:
        area = cv2.contourArea(square)
        if 2500 < area < 5000:
            cv2.polylines(img_warped, [square], True, (0, 255, 0), 2)  # Can be removed in production
            coordinates = [tuple(point) for point in square.tolist()]
            # Sort points by sum and difference of coordinates to order corners
            sorted_coords = sorted(coordinates, key=lambda pt: (pt[0] + pt[1], pt[0] - pt[1]))
            sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]  # Swap BR and TR
            sq_coords.append(sorted_coords)

    print(len(sq_coords))  # Debug: number of detected squares

    # Initialize board dictionary with all squares set to None
    board = {
        f"{col}{row}": None
        for row in range(8, 0, -1)
        for col in 'abcdefgh'
    }
    keys = list(board.keys())

    # Helper function to sort square coordinates consistently
    def sort_coordinates(coords):
        sorted_coords = sorted(coords, key=lambda pt: (pt[0] + pt[1], pt[0] - pt[1]))
        sorted_coords[2], sorted_coords[3] = sorted_coords[3], sorted_coords[2]
        return sorted_coords

    # Sort squares first by vertical (y) and then horizontal (x) coordinate of top-left corner
    sorted_sq_coords = sorted(sq_coords, key=lambda coords: (coords[0][1], coords[0][0]))

    # Group sorted squares into rows of 8 squares each
    rows = [sorted_sq_coords[i:i+8] for i in range(0, len(sorted_sq_coords), 8)]

    # Sort each row left to right
    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda coords: coords[0][0])

    # Flatten rows back into a single list
    sorted_sq_coords = [coords for row in rows for coords in row]

    # Assign sorted square coordinates to board dictionary keys
    for i, key in enumerate(keys):
        if i < len(sorted_sq_coords):
            board[key] = sort_coordinates(sorted_sq_coords[i])

    return img_warped, board


# ----------------------------------------
# Save images of each square in different rotations
# ----------------------------------------
def salvaFoto(board, img_warped, square_name, file_number, folder):
    x1, y1 = board[square_name][0]
    x2, y2 = board[square_name][2]

    # Crop the square from the warped image and resize
    square = img_warped[y1:y2, x1:x2]
    square = cv2.resize(square, (150, 150))

    try:
        mkdir(folder)  # Create folder if it doesn't exist
    except FileExistsError:
        pass

    # Save the square image in 4 rotations
    for i in range(4):
        filename = f"{folder}/{file_number + i}.png"
        cv2.imwrite(filename, square)
        square = cv2.rotate(square, cv2.ROTATE_90_CLOCKWISE)

    print(f"Saved images {file_number} to {file_number + 3} in folder '{folder}'")


# ----------------------------------------
# Main loop to capture video, detect board, warp perspective, and save squares on mouse click
# ----------------------------------------
def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Open video capture device
    end_loop = False
    perspective_matrix = None
    key = None
    board = {}
    folder = "bb"  # Folder to save pieces
    file_number = 1

    while not end_loop:
        ret, img = cap.read()
        if not ret or img is None:
            continue

        img_copy = cv2.resize(img.copy(), (600, 600))

        # When ENTER (key 13) pressed, calculate the warp perspective around chessboard
        if perspective_matrix is None and key == 13:
            perspective_matrix = warpBoard(img_copy)
            img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))
            mask, board = setSquares(img_warped)
            cv2.imshow("ChessBoard", img_warped)
            cv2.imshow("Boxes", mask)

        elif perspective_matrix is not None:
            img_warped = cv2.warpPerspective(img_copy, perspective_matrix, (600, 600))

            # Mouse callback to detect which square is clicked and save its images
            def onMouse(event, x, y, flags, param):
                nonlocal file_number
                if event == cv2.EVENT_LBUTTONDOWN:
                    for square_name, coords in board.items():
                        if coords[0] is None:
                            continue
                        top_left = coords[0]
                        bottom_right = coords[2]
                        if top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]:
                            salvaFoto(board, img_warped, square_name, file_number, folder)
                            file_number += 4
                            return

            cv2.setMouseCallback("ChessBoard", onMouse)
            cv2.imshow("ChessBoard", img_warped)

        else:
            # Show original resized image if no perspective matrix yet
            cv2.imshow("ChessBoard", img_copy)

        key = cv2.waitKey(20)
        if key == 27:  # ESC key to exit
            end_loop = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
