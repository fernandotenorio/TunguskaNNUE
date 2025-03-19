import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import gzip
import time
import os
import math
import random
import sys

from sqlite_sampler import ChessDataset
from torch.utils.data import DataLoader
from constants import *

print(f"Device is {DEVICE}.")

np.random.seed(3)
torch.manual_seed(3)
random.seed(3)


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

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
# ------------------------- End NNUE class  ------------------------- #

def get_all_csv_files(root_folder):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, file))
    return csv_files


def save_npz(checkpoint, outfile):
    model = NNUE()
    model.load(checkpoint)
    weights = {key: value.cpu().numpy() for key, value in model.state_dict().items()}
    np.savez(f"{outfile}.npz", **weights)


def validate(model, val_dataloader, criterion):
    model.eval()
    total_loss = 0.0
    batch_fails = 0
    n_batches = len(val_dataloader)

    with torch.no_grad():  # Disable gradient calculations
        for _, (input_white, input_black, side_to_move, batch_evals) in enumerate(val_dataloader):           
            # Forward pass
            prediction = model(input_white, input_black, side_to_move)

            # Apply sigmoid scaling
            scaled_target = torch.sigmoid(batch_evals / SCALE)
            scaled_predicted = torch.sigmoid(prediction / SCALE)

            # Compute loss
            loss = criterion(scaled_predicted, scaled_target)
            total_loss += loss.item()

    # Compute average loss
    average_loss = total_loss/n_batches
    return average_loss


# --- Training Loop ---
def train(model, db_path, rebuild, positions_folder, val_positions_folder, rebuild_val, checkpoint_dir, epochs, batch_size, optimizer, criterion):
    model.train()
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)

    file_paths = get_all_csv_files(positions_folder)
    dataset = ChessDataset(db_path, batch_size, DEVICE, file_paths, rebuild_db=rebuild)
    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)
    num_batches = dataset.num_batches

    # Validation
    val_files = get_all_csv_files(val_positions_folder)
    val_dataset = ChessDataset("validation.db", batch_size, DEVICE, val_files, rebuild_db=rebuild_val)
    val_dataloader = DataLoader(val_dataset, batch_size=None, batch_sampler=None)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0

        for i, (input_white, input_black, side_to_move, batch_evals) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            prediction = model(input_white, input_black, side_to_move)
            scaled_target = torch.sigmoid(batch_evals / SCALE)
            scaled_predicted = torch.sigmoid(prediction / SCALE)

            loss = criterion(scaled_predicted, scaled_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % PROGRESS_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                batches_done = i
                batches_left = num_batches - batches_done
                time_per_batch = elapsed_time / (batches_done + 1)
                eta = time_per_batch * batches_left
                avg_loss = total_loss / (batches_done + 1)

                print(f"Epoch {epoch + 1}/{epochs}, Batch {batches_done}/{num_batches}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s")

        val_loss = validate(model, val_dataloader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.6f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    model = NNUE().to(DEVICE)

    LEARNING_RATE = 0.001
    BATCH_SIZE = 4096 * 4
    EPOCHS = 100
    CHECKPOINT_DIR = "checkpoints-hl-256"
    PROGRESS_INTERVAL = 10 # every n batches
    DB_PATH = "positions.db"
    POS_FOLDER = "positions_train"
    VAL_FOLDER = "positions_val"
    REBUILD = not True
    REBUILD_VAL = not True

    # Load previous checkpoint
    #model.load(os.path.join("checkpoints-full-256", "model_epoch_2.pth"))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train(model, DB_PATH, REBUILD, POS_FOLDER, VAL_FOLDER, REBUILD_VAL, CHECKPOINT_DIR, EPOCHS, BATCH_SIZE, optimizer, criterion)
  