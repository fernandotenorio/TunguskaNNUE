import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import time

# Constants
INPUT_SIZE = 768
HL_SIZE = 256  # Hidden layer size
SCALE = 400
QA = 255
QB = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 128  # Or whatever batch size you prefer
EPOCHS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
VALIDATION_SPLIT = 0.1
CHECKPOINT_DIR = "checkpoints"  # Directory to save checkpoints
PROGRESS_INTERVAL = 100 # Print progress every N batches.


# --- NNUE Network Definition ---
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.accumulator = nn.Linear(INPUT_SIZE, HL_SIZE)
        self.output = nn.Linear(2 * HL_SIZE, 1)  # Accumulator to output

        # Initialize weights (optional, but good practice)
        nn.init.kaiming_normal_(self.accumulator.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output.weight, nonlinearity='linear')
        #nn.init.zeros_(self.accumulator.weight)
        #nn.init.zeros_(self.output.weight)


    def forward(self, x_white, x_black, side_to_move):
        # x_white: Input from white's perspective
        # x_black: Input from black's perspective
        # side_to_move: 0 for white, 1 for black

        white_accumulator = torch.clamp(self.accumulator(x_white), 0, QA)
        black_accumulator = torch.clamp(self.accumulator(x_black), 0, QA)

        # Concatenate based on side to move
        if side_to_move == 0:  # White to move
            combined_accumulator = torch.cat([white_accumulator, black_accumulator], dim=1)
        else:  # Black to move
            combined_accumulator = torch.cat([black_accumulator, white_accumulator], dim=1)

        eval_raw = self.output(combined_accumulator)
        return eval_raw * SCALE / (QA * QB)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


# --- FEN to Input Conversion ---
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


# --- Data Loading and Splitting ---
def load_and_split_data(filename, validation_split):
    fens = []
    evals = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            fens.append(row[0])
            evals.append(int(row[1]))

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

    return train_fens, train_evals, val_fens, val_evals


# --- Training Loop ---
def train(model, train_fens, train_evals, val_fens, val_evals, epochs, batch_size, optimizer, criterion):
    model.train()
    best_val_loss = float('inf')  # Initialize with a very high loss
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create checkpoint directory

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        num_batches = len(train_fens) // batch_size

        for i in range(0, len(train_fens), batch_size):
            batch_fens = train_fens[i:i + batch_size]
            batch_evals = torch.tensor(train_evals[i:i + batch_size], dtype=torch.float32, device=DEVICE).view(-1, 1)

            optimizer.zero_grad()
            loss = 0
            for fen_index, fen in enumerate(batch_fens):
                input_white, input_black, side_to_move = fen_to_input(fen)
                input_white = input_white.unsqueeze(0)
                input_black = input_black.unsqueeze(0)
                prediction = model(input_white, input_black, side_to_move)
                target_value = batch_evals[fen_index].view(1, 1)
                scaled_target = torch.sigmoid(target_value / SCALE)
                scaled_predicted = torch.sigmoid(prediction / SCALE)
                loss += criterion(scaled_predicted, scaled_target)

            loss = loss / len(batch_fens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- Progress Updates (within epoch) ---
            if (i // batch_size) % PROGRESS_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                batches_done = i // batch_size
                batches_left = num_batches - batches_done
                time_per_batch = elapsed_time / (batches_done + 1)  # Avoid division by zero
                eta = time_per_batch * batches_left
                avg_loss = total_loss / (batches_done + 1)

                print(f"Epoch {epoch + 1}/{epochs}, Batch {batches_done}/{num_batches}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s")

        # --- Validation ---
        val_loss = validate(model, val_fens, val_evals, batch_size, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # --- Checkpointing ---
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # --- Early Stopping (Optional) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # You could also save the *best* model here, not just checkpoints
        # else:  # Uncomment for early stopping
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print("Early stopping triggered.")
        #         break


# --- Validation Function ---
def validate(model, val_fens, val_evals, batch_size, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, len(val_fens), batch_size):
            batch_fens = val_fens[i:i + batch_size]
            batch_evals = torch.tensor(val_evals[i:i + batch_size], dtype=torch.float32, device=DEVICE).view(-1, 1)
            loss = 0
            for fen_index, fen in enumerate(batch_fens):
                input_white, input_black, side_to_move = fen_to_input(fen)
                input_white = input_white.unsqueeze(0)  # Add batch dimension
                input_black = input_black.unsqueeze(0)
                prediction = model(input_white, input_black, side_to_move)
                target_value = batch_evals[fen_index].view(1, 1)
                scaled_target = torch.sigmoid(target_value / SCALE)
                scaled_predicted = torch.sigmoid(prediction / SCALE)
                loss += criterion(scaled_predicted, scaled_target)

            loss = loss/len(batch_fens)
            total_loss += loss.item()
    return total_loss / (len(val_fens) // batch_size)


# --- Evaluation (Example - Same as before) ---
def evaluate_position(model, fen):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        input_white, input_black, side_to_move = fen_to_input(fen)
        input_white = input_white.unsqueeze(0)
        input_black = input_black.unsqueeze(0)
        prediction = model(input_white, input_black, side_to_move)
    return prediction.item()

# --- Main ---
if __name__ == '__main__':
    model = NNUE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Load and split data
    train_fens, train_evals, val_fens, val_evals = load_and_split_data("../NNUE/data/train.csv", VALIDATION_SPLIT)

    # Train the model
    train(model, train_fens, train_evals, val_fens, val_evals, EPOCHS, BATCH_SIZE, optimizer, criterion)

    # --- Example Evaluation ---
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    evaluation = evaluate_position(model, test_fen)
    print(f"Evaluation of {test_fen}: {evaluation}")
