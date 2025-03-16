import torch

INPUT_SIZE = 768
HL_SIZE = 256
SCALE = 400
QA = 255
QB = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# !! IMPORTANT: using WHITE=1, BLACK=0

PIECE_MAP = {
    'p': (0, 0), 'n': (1, 0), 'b': (2, 0), 'r': (3, 0), 'q': (4, 0), 'k': (5, 0),
    'P': (0, 1), 'N': (1, 1), 'B': (2, 1), 'R': (3, 1), 'Q': (4, 1), 'K': (5, 1)
}
