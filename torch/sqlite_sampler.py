import sqlite3
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import glob
import math
import numpy as np

from constants import INPUT_SIZE, DEVICE
from dataloader import fen_to_input

class ChessDataset(Dataset):
    def __init__(self, db_path, batch_size, device, file_paths=None, rebuild_db=False):
        """
        Args:
            db_path (str): Path to the SQLite database.
            batch_size (int): Size of each batch.
            file_paths (list, optional): List of CSV file paths. Only needed if rebuilding the DB.
            rebuild_db (bool): Whether to rebuild the database from CSV files.
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.device = device
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if rebuild_db:
            if file_paths is None:
                raise ValueError("file_paths must be provided if rebuild_db is True")
            self._create_and_populate_db(file_paths)

        # Get the total number of rows
        self.cursor.execute("SELECT COUNT(*) FROM chess_data")
        self.total_rows = self.cursor.fetchone()[0]
        print(f"Total rows: {self.total_rows}")

        # Calculate the number of batches (handling non-divisible cases)
        self.num_batches = math.ceil(self.total_rows / self.batch_size)

        self.current_epoch = 0
        self.batch_indices = self._generate_batch_indices()


    def _create_and_populate_db(self, file_paths):
        """Creates the database and populates it from CSV files."""
        print("Creating and populating database...")
        self.cursor.execute("DROP TABLE IF EXISTS chess_data")
        self.cursor.execute("""
            CREATE TABLE chess_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                FEN TEXT,
                Evaluation REAL
            )
        """)
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            try:
                df = pd.read_csv(file_path)
                if 'FEN' not in df.columns or 'Evaluation' not in df.columns:
                    print(f"Skipping {file_path} due to missing columns.")
                    continue
                df = df[['FEN', 'Evaluation']]
                df.to_sql('chess_data', self.conn, if_exists='append', index=False)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Skipping {file_path} due to error: {e}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        self.conn.commit()
        print("Database created and populated.")

    def _generate_batch_indices(self):
        """Generates a list of indices for each batch for the current epoch."""
        all_indices = list(range(self.total_rows))
        random.shuffle(all_indices)
        batch_indices = []
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.total_rows)
            batch_indices.append(all_indices[start_idx:end_idx])
        return batch_indices

    def __len__(self):
        """Returns the number of batches."""
        return self.num_batches

    def __getitem__(self, idx):
        """Retrieves a batch of data using a temporary table."""
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")

        if idx == 0 and self.current_epoch > 0:
            self.batch_indices = self._generate_batch_indices()

        batch_indices = self.batch_indices[idx]
        batch_size = len(batch_indices)

        self.cursor.execute("DROP TABLE IF EXISTS temp_batch_ids")
        self.cursor.execute("CREATE TEMP TABLE temp_batch_ids (id INTEGER)")
        self.cursor.executemany("INSERT INTO temp_batch_ids (id) VALUES (?)", [(x + 1,) for x in batch_indices])
        self.conn.commit()

        query = """
            SELECT chess_data.FEN, chess_data.Evaluation
            FROM chess_data
            INNER JOIN temp_batch_ids ON chess_data.id = temp_batch_ids.id
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        self.cursor.execute("DROP TABLE temp_batch_ids")

        positions, evaluations = zip(*results)

        # Initialize NumPy arrays *before* the loop
        white_arrays = np.zeros((batch_size, INPUT_SIZE), dtype=np.float32)
        black_arrays = np.zeros((batch_size, INPUT_SIZE), dtype=np.float32)
        stm_arrays = np.zeros((batch_size, 1), dtype=np.float32)

        for i, pos in enumerate(positions):
            white, black, stm = self.position_to_tensor(pos)          
            white_arrays[i] = white
            black_arrays[i] = black
            stm_arrays[i] = stm

        # Convert to PyTorch tensors *after* the loop
        batch_white = torch.tensor(white_arrays, device=self.device)
        batch_black = torch.tensor(black_arrays, device=self.device)
        batch_stm = torch.tensor(stm_arrays, device=self.device)
        batch_evals = torch.tensor(evaluations, device=self.device, dtype=torch.float32).unsqueeze(1)

        if idx == self.num_batches - 1:
            self.current_epoch += 1

        return batch_white, batch_black, batch_stm, batch_evals

   
    def position_to_tensor(self, position_str):
        board_tensor_white, board_tensor_black, side_to_move = fen_to_input(position_str, INPUT_SIZE)
        # white_tensor = torch.tensor(board_tensor_white, dtype=torch.float32, device=self.device)
        # black_tensor = torch.tensor(board_tensor_black, dtype=torch.float32, device=self.device)
        # stm_tensor = torch.tensor([side_to_move], dtype=torch.float32, device=self.device)  # Keep as float
        # return white_tensor, black_tensor, stm_tensor
        return board_tensor_white, board_tensor_black, side_to_move
       

    def close(self):
        self.conn.close()


# --- Example Usage ---
if __name__ == '__main__':
    db_path = 'test.db'
    csv_dir = 'positions_val'
    batch_size = 16  # Large batch size
    num_epochs = 10

    file_paths = glob.glob(os.path.join(csv_dir, '*.csv'))

    # --- Create Dataset and DataLoader ---
    dataset = ChessDataset(db_path, batch_size, DEVICE, file_paths, rebuild_db=True)
    # batch_size=None and batch_sampler=None are crucial here!
    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

    for epoch in range(num_epochs):
        for batch_idx, (input_white, input_black, side_to_move, evaluations) in enumerate(dataloader):
            print(input_white.shape, input_black.shape, side_to_move.shape, evaluations.shape)

    dataset.close()
