import sqlite3
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import glob
import math

class ChessDataset(Dataset):
    def __init__(self, db_path, batch_size, file_paths=None, rebuild_db=False):
        """
        Args:
            db_path (str): Path to the SQLite database.
            batch_size (int): Size of each batch.
            file_paths (list, optional): List of CSV file paths. Only needed if rebuilding the DB.
            rebuild_db (bool): Whether to rebuild the database from CSV files.
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if rebuild_db:
            if file_paths is None:
                raise ValueError("file_paths must be provided if rebuild_db is True")
            self._create_and_populate_db(file_paths)

        # Get the total number of rows
        self.cursor.execute("SELECT COUNT(*) FROM chess_data")
        self.total_rows = self.cursor.fetchone()[0]

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
                position TEXT,
                evaluation REAL
            )
        """)
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            try:
                df = pd.read_csv(file_path)
                if 'position' not in df.columns or 'evaluation' not in df.columns:
                    print(f"Skipping {file_path} due to missing columns.")
                    continue
                df = df[['position', 'evaluation']]
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

        self.cursor.execute("DROP TABLE IF EXISTS temp_batch_ids")
        self.cursor.execute("CREATE TEMP TABLE temp_batch_ids (id INTEGER)")
        self.cursor.executemany("INSERT INTO temp_batch_ids (id) VALUES (?)", [(x + 1,) for x in batch_indices])
        self.conn.commit()

        query = """
            SELECT chess_data.position, chess_data.evaluation
            FROM chess_data
            INNER JOIN temp_batch_ids ON chess_data.id = temp_batch_ids.id
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        self.cursor.execute("DROP TABLE temp_batch_ids")

        positions, evaluations = zip(*results)
        positions_tensor = torch.tensor([self.position_to_tensor(pos) for pos in positions], dtype=torch.float32)
        evaluations_tensor = torch.tensor(evaluations, dtype=torch.float32).unsqueeze(1)

        if idx == self.num_batches - 1:
          self.current_epoch += 1

        return positions_tensor, evaluations_tensor

    def position_to_tensor(self, position_str):
        """Converts a chess position string to a tensor."""
        piece_map = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                     'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12, '.': 0}
        tensor = torch.zeros(8 * 8 * 13)
        board_list = position_str.split(" ")[0].split("/")
        for row_idx, row_str in enumerate(board_list):
            col_idx = 0
            for char in row_str:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    piece_type = piece_map[char]
                    tensor_idx = row_idx * (8 * 8 * 13) + col_idx * 13 + piece_type
                    tensor[tensor_idx] = 1
                    col_idx += 1
        return tensor

    def close(self):
        self.conn.close()


# --- Example Usage ---
if __name__ == '__main__':
    db_path = 'chess_data.db'
    csv_dir = './csv_files'
    batch_size = 10000  # Large batch size
    num_epochs = 10
    learning_rate = 0.001

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        num_dummy_files = 5
        rows_per_file = 25003  # Use a non-multiple of batch_size
        for i in range(num_dummy_files):
            dummy_data = {
                'position': ['.' * 8 + '/' + '.' * 8 + '/' + '.' * 8 + '/' + '..p.....' + '/' + '........' + '/' + '........' + '/' + 'PPPPPPPP' + '/' + 'RNBQKBNR' + ' w - - 0 1'] * rows_per_file,
                'evaluation': [random.uniform(-1, 1) for _ in range(rows_per_file)]
            }
            dummy_df = pd.DataFrame(dummy_data)
            dummy_df.to_csv(os.path.join(csv_dir, f'dummy_data_{i}.csv'), index=False)

    file_paths = glob.glob(os.path.join(csv_dir, '*.csv'))

    # --- Create Dataset and DataLoader ---
    dataset = ChessDataset(db_path, batch_size, file_paths, rebuild_db=True)
    # batch_size=None and batch_sampler=None are crucial here!
    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

    # --- Define a Simple Model ---
    class SimpleChessNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(8 * 8 * 13, 128)
            self.linear2 = torch.nn.Linear(128, 64)
            self.linear3 = torch.nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    model = SimpleChessNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (positions, evaluations) in enumerate(dataloader):
            # No need to manually manage batch indices here
            optimizer.zero_grad()
            outputs = model(positions)
            loss = criterion(outputs, evaluations)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Print loss per batch
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(dataloader):.4f}')

    dataset.close()
    print("Training complete.")