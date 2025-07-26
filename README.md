# ChessboardVision 🧠♟️

A project developed by two friends passionate about chess and AI: a system to **digitize real chess games using only a webcam**, Computer Vision, and a custom-trained neural network.

## 🎯 Objective

The goal is to provide a **low-cost, accessible alternative** to commercial electronic chessboards (which can cost €800+), allowing any chess enthusiast to record and analyze their games with just:

- A regular chessboard and pieces
- A webcam (even a smartphone)
- Our software

## 🛠️ Features

- Real-time **chessboard detection** using OpenCV
- **Piece recognition** via a custom CNN trained on a hand-made dataset
- **Move detection** through motion tracking
- Support for **special moves**: castling, en passant, promotion
- **Automatic FEN generation** for every board state
- **Digital replay** of the match with navigation
- Export of the full game for later analysis

## 🧠 AI and Model Training

We created our own dataset of over **3,500 labeled images** of each piece and square.  
We trained a CNN using PyTorch, achieving **91.3% test accuracy**, using:
- AdamW optimizer
- CrossEntropyLoss
- Data augmentation (rotation, brightness, color)
- 250 epochs on Google Colab (CUDA)

## 🧪 Limitations

- Detection may fail under poor lighting or unstable camera conditions
- Motion detection can occasionally misfire if the camera shakes
- Initial setup only supports standard starting positions (not custom ones)
- Minor inaccuracies may occur with visually similar pieces (e.g. black knights)

## ▶️ Demo Video

Watch a short demonstration of the program in action:  
📺 [https://youtu.be/HkP07RBLzWU](https://youtu.be/HkP07RBLzWU)

## 💻 How It Works

The program (`chessboardVision.py`) operates in three phases:

1. **Board Setup**  
   Detect the board via perspective transformation, identify all 64 squares, and recognize starting positions using the trained CNN.

2. **Game Tracking**  
   Detect piece movements with motion detection and update the virtual board state accordingly, handling all standard chess rules.

3. **Game Replay & Save**  
   Once the game ends, you can navigate through the moves using your keyboard and save the game in FEN format.

## 📦 Requirements

Install the following libraries before running:

- `opencv-python`
- `numpy`
- `torch` + `torchvision`
- `Pillow`
- `chess-board`
- `python-chess`

You can install them via:

```bash
pip install opencv-python numpy torch torchvision pillow chess-board python-chess
