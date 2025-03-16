import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import gc
import numpy as np
from constants import DEVICE, PIECE_MAP


def fen_to_input(fen, inp_size):
    board_tensor_white = np.zeros(inp_size)
    board_tensor_black = np.zeros(inp_size)

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
            piece_type, piece_color = PIECE_MAP[char]
            square = rank * 8 + file
            
            # Calculate indices for white and black perspectives
            index_white = piece_color * 64 * 6 + piece_type * 64 + square
            index_black = (1-piece_color) * 64 * 6 + piece_type * 64 + (square ^ 0b111000)

            board_tensor_white[index_white] = 1
            board_tensor_black[index_black] = 1
            file += 1
    return board_tensor_white, board_tensor_black, side_to_move


# def collate_fn(batch):
#     # Unzip the batch into input components (this is a list of tuples)
#     input_white, input_black, side_to_move, eval_values = zip(*batch)
#     input_white = np.array(input_white)
#     input_black = np.array(input_black)
    
#     # Convert lists of inputs into tensors
#     input_white = torch.tensor(input_white, dtype=torch.float32, device=DEVICE)  # Shape: (batch_size, input_size)
#     input_black = torch.tensor(input_black, dtype=torch.float32, device=DEVICE)
#     side_to_move = torch.tensor(side_to_move, dtype=torch.int64, device=DEVICE)  # Shape: (batch_size,)
    
#     # Convert eval values to tensor (we assume eval_values is a list of floats)
#     eval_tensor = torch.tensor(eval_values, dtype=torch.float32, device=DEVICE).view(-1, 1)  # Shape: (batch_size, 1)
#     return input_white, input_black, side_to_move, eval_tensor

# TODO pickle tensor?
class FenDataset(Dataset):
    def __init__(self, folder_path, inp_size):
        self.folder_path = folder_path
        self.inp_size = inp_size
        self.file_list = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path) if "_part_" in f and f.endswith(".csv")]
        )

        self.data = []
        for file_path in self.file_list:
            try:
                df = pd.read_csv(file_path)
                fens_evals = df.values.tolist()
                del df
                gc.collect()

                for d in fens_evals:
                    w_inp, b_inp, stm = fen_to_input(d[0], self.inp_size)
                    self.data.append([
                        torch.tensor(w_inp, dtype=torch.float32, device=DEVICE),
                        torch.tensor(b_inp, dtype=torch.float32, device=DEVICE),
                        torch.tensor(stm, dtype=torch.float32, device=DEVICE),
                        torch.tensor(d[1], dtype=torch.float32, device=DEVICE),
                    ])
                del fens_evals
                gc.collect()
                print(f"Done creating tensors for '{file_path}'")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        self.total_items = len(self.data)
        print(f"Loaded {self.total_items} samples.")
        gc.collect()

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        # fen_position, eval_value = self.data[idx]
        # input_white, input_black, side_to_move = fen_to_input(fen_position, self.inp_size)
        # return input_white, input_black, side_to_move, eval_value
        return self.data[idx]
