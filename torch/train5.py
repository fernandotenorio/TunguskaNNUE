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

from dataloader import FenDataset, collate_fn
from torch.utils.data import Dataset, DataLoader, random_split
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
# ------------------------- End NNUE class  ------------------------- #

def load_batch(file):
    with gzip.open(file, 'rb') as f:
       white, black, stm, evals = torch.load(f)
       white = white.to(DEVICE)
       black = black.to(DEVICE)
       stm = stm.to(DEVICE)
       evals = evals.to(DEVICE).view(-1, 1)
       return white, black, stm, evals

def split_list(data, val_ratio):
    assert 0 <= val_ratio <= 1
    train_ratio = 1 - val_ratio
    data_copy = data.copy()
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    val_data = data_copy[split_idx:]
    return train_data, val_data

def validate(model, val_batches, criterion):
    model.eval()
    total_loss = 0.0
    batch_fails = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_file in val_batches:
            try:
                input_white, input_black, side_to_move, eval_tensor = load_batch(batch_file)
            except Exception as e:
                print(f'Validation: Failed to load file {batch_file}.')
                batch_fails+= 1
                continue

            # Forward pass
            prediction = model(input_white, input_black, side_to_move)

            # Apply sigmoid scaling
            scaled_target = torch.sigmoid(eval_tensor / SCALE)
            scaled_predicted = torch.sigmoid(prediction / SCALE)

            # Compute loss
            loss = criterion(scaled_predicted, scaled_target)
            total_loss += loss.item()

    # Compute average loss
    average_loss = total_loss/(len(val_batches) - batch_fails)
    return average_loss


# --- Training Loop ---
def train(model, batches_folder, checkpoint_dir, epochs, batch_size, val_split, optimizer, criterion):
    model.train()
    best_val_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # only one batches folder for now
    batch_epoch_folder = os.path.join(batches_folder, f'epoch_{0}')
    batch_files = [os.path.join(batch_epoch_folder, f) for f in os.listdir(batch_epoch_folder) if f.endswith('.pt.gz')]
    train_batches, val_batches = split_list(batch_files, val_split)
    print(f"Train batches: {len(train_batches)} / Validation batches: {len(val_batches)}")

    for epoch in range(epochs):
        random.shuffle(train_batches)

        start_time = time.time()
        total_loss = 0
        num_batches = len(train_batches)
        batch_fails = 0

        for i, batch_file in enumerate(train_batches):
            try:
                input_white, input_black, side_to_move, batch_evals = load_batch(batch_file)
            except Exception as e:
                print(f'Train: Failed to load file {batch_file}.')
                batch_fails+= 1
                # save model just in case
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
                model.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                continue
            
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
                avg_loss = total_loss / (batches_done + 1 - batch_fails)

                print(f"Epoch {epoch + 1}/{epochs}, Batch {batches_done}/{num_batches}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"Elapsed: {elapsed_time:.2f}s, ETA: {eta:.2f}s")

        val_loss = validate(model, val_batches, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.6f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch + 1}.pth")
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"Best val loss: {best_val_loss:.6f}")


def save_npz(checkpoint, outfile):
    model = NNUE()
    model.load(checkpoint)
    weights = {key: value.cpu().numpy() for key, value in model.state_dict().items()}
    np.savez(f"{outfile}.npz", **weights)


if __name__ == '__main__':
    model = NNUE().to(DEVICE)

    BATCHES_FOLDER = "batches"
    LEARNING_RATE = 0.001
    BATCH_SIZE = None#2 * 4096
    EPOCHS = 100
    VALIDATION_SPLIT = 0.1
    CHECKPOINT_DIR = "checkpoints-full-256"
    PROGRESS_INTERVAL = 50 # every n batches

    # Load previous checkpoint
    model.load(os.path.join("checkpoints-full-256", "model_epoch_2.pth"))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train(model, BATCHES_FOLDER, CHECKPOINT_DIR, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, optimizer, criterion)
  