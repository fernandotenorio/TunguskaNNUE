import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import time
import os
import math
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Constants
INPUT_SIZE = 768
HL_SIZE = 64
SCALE = 400
QA = 255
QB = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 512
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VALIDATION_SPLIT = 0.1
CHECKPOINT_DIR = "test_val_tmp"#"checkpoints-02-feb-pt2"
#DATASET = "../raw_data/02-feb-parts/02-feb_part_2.csv"
DATASET = "../raw_data/01-jan.csv"
PROGRESS_INTERVAL = 100 # every n batches


# --- NNUE Network Definition ---
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.accumulator = nn.Linear(INPUT_SIZE, HL_SIZE)
        self.output_weights = nn.Parameter(torch.randn(2 * HL_SIZE, 1))  # Store output weights directly
        self.output_bias = nn.Parameter(torch.randn(1))  # Store output bias directly
        # Initialize weights
        nn.init.kaiming_normal_(self.accumulator.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_weights, nonlinearity='linear') # Corrected initialization

    def forward_old(self, x_white, x_black, side_to_move):
        # Training forward pass (still uses nn.Linear for backpropagation)
        white_accumulator = torch.clamp(self.accumulator(x_white), 0, QA)
        black_accumulator = torch.clamp(self.accumulator(x_black), 0, QA)

        if side_to_move == 0:
            combined_accumulator = torch.cat([white_accumulator, black_accumulator], dim=1)
        else:
            combined_accumulator = torch.cat([black_accumulator, white_accumulator], dim=1)

        # Use manual dot product during training (for consistency with inference)
        eval_raw = torch.matmul(combined_accumulator, self.output_weights) + self.output_bias
        return eval_raw * SCALE / (QA * QB)

    def forward(self, x_white, x_black, side_to_move):
        # Compute accumulator for both white and black in parallel
        white_accumulator = torch.clamp(self.accumulator(x_white), 0, QA)
        black_accumulator = torch.clamp(self.accumulator(x_black), 0, QA)

        # Select accumulator order based on side to move (batch-wise operation)
        combined_accumulator = torch.where(
            side_to_move.view(-1, 1).bool(),
            torch.cat([black_accumulator, white_accumulator], dim=1),
            torch.cat([white_accumulator, black_accumulator], dim=1),
        )

        eval_raw = combined_accumulator @ self.output_weights + self.output_bias
        return eval_raw * SCALE / (QA * QB)

    def inference(self, stm_accumulator, nstm_accumulator):
        # Efficient inference (dot product only)
        combined_accumulator = torch.cat([stm_accumulator, nstm_accumulator], dim=0) # No batch dimension
        eval_raw = torch.dot(combined_accumulator, self.output_weights.squeeze()) + self.output_bias
        #eval_raw = torch.matmul(combined_accumulator.unsqueeze(0), self.output_weights) + self.output_bias #chatgpt: better for batch
        return eval_raw * SCALE / (QA * QB)

    def accumulator_add(self, accumulator, index):
        # Efficiently add to the accumulator
        #accumulator = accumulator + self.accumulator.weight[:, index].clone()  # Corrected indexing
        accumulator.add_(self.accumulator.weight[:, index])
        return accumulator

    def accumulator_sub(self, accumulator, index):
        # Efficiently subtract from the accumulator
        #accumulator = accumulator - self.accumulator.weight[:, index].clone()  # Corrected indexing
        accumulator.sub_(self.accumulator.weight[:, index])  # Corrected indexing
        return accumulator

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


# --- FEN to Input Conversion (Modified for Efficient Updates) ---
def fen_to_input(fen):
    piece_map = {
        'p': (0, 0), 'n': (1, 0), 'b': (2, 0), 'r': (3, 0), 'q': (4, 0), 'k': (5, 0),
        'P': (0, 1), 'N': (1, 1), 'B': (2, 1), 'R': (3, 1), 'Q': (4, 1), 'K': (5, 1)
    }
    board_tensor_white = torch.zeros(INPUT_SIZE, device=DEVICE)
    board_tensor_black = torch.zeros(INPUT_SIZE, device=DEVICE)

    parts = fen.split()
    board_str = parts[0]
    side_to_move = 0 if parts[1] == 'w' else 1

    rank = 7
    file = 0
    for char in board_str:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            piece_type, piece_color = piece_map[char]
            square = rank * 8 + file
            
            # Calculate indices for white and black perspectives
            index_white = piece_color * 64 * 6 + piece_type * 64 + square
            index_black = (1-piece_color) * 64 * 6 + piece_type * 64 + (square ^ 0b111000)

            board_tensor_white[index_white] = 1
            board_tensor_black[index_black] = 1
            file += 1

    return board_tensor_white, board_tensor_black, side_to_move


def fen_to_accumulator(model, fen):
    # Initialize accumulators
    white_accumulator = torch.zeros(HL_SIZE, device=DEVICE) # Put on correct device
    black_accumulator = torch.zeros(HL_SIZE, device=DEVICE) # Put on correct device

    piece_map = {
        'p': (0, 0), 'n': (1, 0), 'b': (2, 0), 'r': (3, 0), 'q': (4, 0), 'k': (5, 0),
        'P': (0, 1), 'N': (1, 1), 'B': (2, 1), 'R': (3, 1), 'Q': (4, 1), 'K': (5, 1)
    }
    parts = fen.split()
    board_str = parts[0]
    side_to_move = 0 if parts[1] == 'w' else 1

    rank = 7;
    file = 0;
    for char in board_str:
        if char == '/':
            rank -= 1;
            file = 0;
        elif char.isdigit():
            file += int(char);
        else:
            piece_type, piece_color = piece_map[char];
            square = rank * 8 + file;

            index_white = piece_color * 64 * 6 + piece_type * 64 + square;
            index_black = (1 - piece_color) * 64 * 6 + piece_type * 64 + (square ^ 0b111000);

            white_accumulator = model.accumulator_add(white_accumulator, index_white);
            black_accumulator = model.accumulator_add(black_accumulator, index_black);
            file += 1;

    return white_accumulator, black_accumulator, side_to_move

def apply_move(model, white_accumulator, black_accumulator, prev_fen, new_fen):
    """Applies a move and updates the accumulators efficiently."""
    prev_white, prev_black, _ = fen_to_input(prev_fen)
    new_white, new_black, side_to_move = fen_to_input(new_fen)

    # Find the differences (pieces that were added or removed)
    diff_white = new_white - prev_white
    diff_black = new_black - prev_black

    # Update accumulators based on the differences
    for i in range(INPUT_SIZE):
        if diff_white[i] > 0:  # Piece added for white's perspective
            white_accumulator = model.accumulator_add(white_accumulator, i)
        elif diff_white[i] < 0:  # Piece removed for white's perspective
            white_accumulator = model.accumulator_sub(white_accumulator, i)

        if diff_black[i] > 0:
            black_accumulator = model.accumulator_add(black_accumulator, i)
        elif diff_black[i] < 0:
            black_accumulator = model.accumulator_sub(black_accumulator, i)
    if side_to_move == 0:
        return white_accumulator, black_accumulator, side_to_move
    else:
        return black_accumulator, white_accumulator, side_to_move


# --- Data Loading and Splitting (Same as before) ---
def load_and_split_data(filename, validation_split):
    fens = []
    evals = []
    eval_correction = {'w': 1, 'b': 1} #accumulator [white black] or [black white], eval on file is already relative to side.

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            fens.append(row[0])
            stm = row[0].split()[1]
            evals.append(int(row[1]) * eval_correction[stm])

    # Shuffle the data
    combined = list(zip(fens, evals))
    np.random.shuffle(combined)
    fens[:], evals[:] = zip(*combined)

    # Split into training and validation sets
    split_index = int(len(fens) * (1 - validation_split))
    train_fens = fens[:split_index]
    train_evals = evals[:split_index]
    val_fens = fens[split_index:]
    val_evals = evals[split_index:]

    print('Done loading and spliting data')
    return train_fens, train_evals, val_fens, val_evals


# --- Training Loop ---
def train(model, train_fens, train_evals, val_fens, val_evals, epochs, batch_size, optimizer, criterion):
    model.train()
    best_val_loss = float('inf')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = len(train_fens) // batch_size

        for i in range(0, len(train_fens), batch_size):
            batch_fens = train_fens[i:i + batch_size]
            batch_evals = torch.tensor(train_evals[i:i + batch_size], dtype=torch.float32, device=DEVICE).view(-1, 1)

            # Convert batch FENs to batched tensor inputs
            input_white_list, input_black_list, side_to_move_list = zip(*[fen_to_input(fen) for fen in batch_fens])
            input_white = torch.stack(input_white_list)
            input_black = torch.stack(input_black_list)
            side_to_move = torch.tensor(side_to_move_list, dtype=torch.int64, device=DEVICE)

            optimizer.zero_grad()

            # Forward pass
            prediction = model(input_white, input_black, side_to_move)
            scaled_target = torch.sigmoid(batch_evals / SCALE)
            scaled_predicted = torch.sigmoid(prediction / SCALE)

            loss = criterion(scaled_predicted, scaled_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i // batch_size) % PROGRESS_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                batches_done = i // batch_size
                batches_left = num_batches - batches_done
                time_per_batch = elapsed_time / (batches_done + 1)
                eta = time_per_batch * batches_left
                avg_loss = total_loss / (batches_done + 1)

                print(f"Epoch {epoch + 1}/{epochs}, Batch {batches_done}/{num_batches}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s")

        val_loss = validate(model, val_fens, val_evals, batch_size, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss


# --- Validation Function ---
def validate(model, val_fens, val_evals, batch_size, criterion):
    model.eval()
    total_loss = 0
    num_batches = len(val_fens) // batch_size

    with torch.no_grad():
        for i in range(0, len(val_fens), batch_size):
            batch_fens = val_fens[i:i + batch_size]
            batch_evals = torch.tensor(val_evals[i:i + batch_size], dtype=torch.float32, device=DEVICE).view(-1, 1)

            # Convert batch FENs to batched tensor inputs
            input_white_list, input_black_list, side_to_move_list = zip(*[fen_to_input(fen) for fen in batch_fens])
            input_white = torch.stack(input_white_list)
            input_black = torch.stack(input_black_list)
            side_to_move = torch.tensor(side_to_move_list, dtype=torch.int64, device=DEVICE)

            # Forward pass (batch processing)
            prediction = model(input_white, input_black, side_to_move)

            # Apply sigmoid scaling
            scaled_target = torch.sigmoid(batch_evals / SCALE)
            scaled_predicted = torch.sigmoid(prediction / SCALE)

            # Compute loss for the entire batch
            loss = criterion(scaled_predicted, scaled_target)
            total_loss += loss.item()

    return total_loss / num_batches


# This resets the accumulators
# --- Evaluation (Example - Now with Efficient Updates) ---
def evaluate_position(model, fen):
    model.eval()
    with torch.no_grad():
        white_accumulator, black_accumulator, side_to_move = fen_to_accumulator(model, fen)
        if side_to_move == 0:
            prediction = model.inference(white_accumulator, black_accumulator)
        else:
            prediction = model.inference(black_accumulator, white_accumulator)

    return prediction.item()


def win_probability_to_cp(wp, k=1/SCALE):
    # Avoid division by zero or log of zero
    wp = min(max(wp, 0.0000001), 0.9999999)
    return -math.log((1 / wp) - 1) / k


# --- Main ---
if __name__ == '__main__':
    model = NNUE().to(DEVICE)

    # Load previous checkpoint
    #model.load(os.path.join("checkpoints-02-feb-pt1", "model_epoch_4.pth"))
    model.load(os.path.join("checkpoints-02-feb-pt1", "model_epoch_4.pth"))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_fens, train_evals, val_fens, val_evals = load_and_split_data(DATASET, VALIDATION_SPLIT)
    train(model, train_fens, train_evals, val_fens, val_evals, EPOCHS, BATCH_SIZE, optimizer, criterion)

    # --- Example Evaluation with Efficient Updates ---
    n = 10
    err = 0

    for i in range(n):
        target_cp = val_evals[i]
        ev = evaluate_position(model, val_fens[i])
        print(f"FEN: {val_fens[i]}")
        print(f"Target (centipawns): {target_cp}")
        print(f"Prediction (centipawns): {ev:.0f}")  # Format to integer centipawns
        print(f"Difference: {target_cp - ev:.0f}")  # Show the difference
        err+= (target_cp - ev)**2
        print("---")

    print(f"RMSE: {(err/n)**0.5}")


    seq_fens = [
                ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "startpos"),
                ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e4"),
                ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e5"),
                ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR b KQkq - 1 2", "Qf3"),
                ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 2 3", "Nc6"),
                ("r1bqkbnr/pppp1Qpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3", "Qxf6"),
                ("r1bq1bnr/pppp1kpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR w KQ - 0 4", "xf6"),
    ]

    white_acc, black_acc, side_to_move = fen_to_accumulator(model, seq_fens[0][0])
    initial_eval = model.inference(white_acc, black_acc)
    print(f"Evaluation of start position: {initial_eval.item()}")

    for i in range(1, len(seq_fens)):
        if i % 2 == 0:
            stm = white_acc
            n_stm = black_acc
        else:
            stm = black_acc
            n_stm = white_acc

        white_acc, black_acc, side_to_move = apply_move(model, white_acc, black_acc, seq_fens[i - 1][0], seq_fens[i][0])
        new_eval = model.inference(stm, n_stm)
        print(f"Evaluation after {seq_fens[i][1]}: {new_eval.item()}")
