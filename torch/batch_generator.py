import os
import torch
import pandas as pd
import numpy as np
import random
import gzip
import pickle
from itertools import zip_longest
from dataloader import fen_to_input

def mix_batches(batch_tuples, new_batch_size=None, keep_remainder=True):
    """
    Mix a list of batch tuples into new batches.
    
    Parameters:
    batch_tuples: List of tuples, where each tuple contains tensors of shape (batch, dim)
    new_batch_size: Size of the new batches. If None, will use the same size as the first batch.
    keep_remainder: If True, include the remaining samples that don't make a full batch
    
    Returns:
    List of mixed batch tuples with the same structure as the input tuples
    """
    # Check if the list is empty
    if not batch_tuples:
        return []
    
    # Get the number of elements in each tuple
    num_elements = len(batch_tuples[0])
    
    # Collect all data for each position in the tuple
    all_data = [[] for _ in range(num_elements)]
    
    for batch_tuple in batch_tuples:
        for i, tensor in enumerate(batch_tuple):
            all_data[i].append(tensor)
    
    # Concatenate each position's data
    concatenated_data = []
    for position_data in all_data:
        concatenated_data.append(torch.cat(position_data, dim=0))
    
    # Get the total number of samples
    total_samples = concatenated_data[0].size(0)
    
    # If new_batch_size is not specified, use the size of the first batch
    if new_batch_size is None:
        new_batch_size = batch_tuples[0][0].size(0)
    
    # Create a common shuffling index
    shuffle_indices = torch.randperm(total_samples)
    
    # Apply the same shuffling to all tensors
    shuffled_data = [data[shuffle_indices] for data in concatenated_data]
    
    # Calculate how many complete new batches we can form
    num_complete_batches = total_samples // new_batch_size
    
    # Create the new batches
    mixed_batches = []
    
    # Process complete batches
    for i in range(num_complete_batches):
        start_idx = i * new_batch_size
        end_idx = start_idx + new_batch_size
        
        # Create a new tuple with the same structure
        new_batch = tuple(data[start_idx:end_idx] for data in shuffled_data)
        mixed_batches.append(new_batch)
    
    # Handle the remainder
    if keep_remainder and total_samples % new_batch_size > 0:
        start_idx = num_complete_batches * new_batch_size
        # Create a final batch with the remaining samples
        remainder_batch = tuple(data[start_idx:] for data in shuffled_data)
        mixed_batches.append(remainder_batch)
    
    return mixed_batches


def save_compressed_tensor(data_list, inp_size, filepath):
    white, black, stm, evals = [], [], [], []

    for fen_position, ev in data_list:
        w, b, side = fen_to_input(fen_position, inp_size)
        white.append(w)
        black.append(b)
        stm.append(side)
        evals.append(ev)

    white = np.array(white, dtype=np.float32)
    black = np.array(black, dtype=np.float32)
    stm = np.array(stm, dtype=int)
    evals = np.array(evals, dtype=int)

    white = torch.tensor(white, dtype=torch.float32, device='cpu')
    black = torch.tensor(black, dtype=torch.float32, device='cpu')
    stm = torch.tensor(stm, dtype=torch.int64, device='cpu')
    evals = torch.tensor(evals, dtype=torch.int64, device='cpu')
    data = white, black, stm, evals

    with gzip.open(filepath, 'wb') as f:
        #pickle.dump(data, f)
        torch.save(data, f)


def mix_files(data_dirs):
    files = []
    for data_dir in data_dirs:
        csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        random.shuffle(csv_files)
        files.append(csv_files)

    if len(files) == 1:
        files.append([])

    files_mix = list(zip_longest(*files, fillvalue=None)) # [(fl1_d1, fl2_d2), (fl2_d1, fl2_d2), ...]
    return files_mix


def create_batches(data_dirs, output_dir, dtype, batch_size, inp_size, n_epochs):
    os.makedirs(output_dir, exist_ok=True)
   
    for epoch in range(n_epochs):
        files_mix = mix_files(data_dirs)
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        batch_idx = 0
        
        for file_tuple in files_mix:
            dfs = []    
            for file in file_tuple:
                if file is None:
                    continue
                df_tmp = pd.read_csv(file, dtype=dtype)
                dfs.append(df_tmp)

            df = pd.concat(dfs)
            df = df.sample(frac=1)
            df_values = df.values.tolist()
            del df
            
            for i in range(0, len(df_values), batch_size):
                batch = df_values[i : i + batch_size]
                batch_file = os.path.join(epoch_dir, f"batch_{batch_idx}.pt.gz")
                save_compressed_tensor(batch, inp_size, batch_file)
                batch_idx += 1

        print(f'Done epoch {epoch + 1} of {n_epochs}.')

        
     

if __name__ == "__main__":
    # create_batches(
    #     data_dirs=["../raw_data/data-d9-parts", "../raw_data/02-feb-parts"],
    #     output_dir="batches",
    #     dtype={'FEN': str, 'Evaluation': int},
    #     batch_size=4096 * 8,
    #     inp_size=768,
    #     n_epochs=10 
    # )
    pass
