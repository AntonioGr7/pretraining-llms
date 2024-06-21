import numpy as np
import torch
from torch.nn import functional as F

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import yaml
import os
import re

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_next_log_file(directory):
    # List all files in the directory
    files_in_directory = os.listdir(directory)

    # Filter files that match the pattern 'log_<number>.txt'
    log_files = [file for file in files_in_directory if re.match(r'^log_\d+\.txt$', file)]

    # Extract the numeric part of the filenames
    log_numbers = [int(re.search(r'(\d+)', file).group()) for file in log_files]

    # Determine the highest number
    if log_numbers:
        max_number = max(log_numbers)
    else:
        max_number = 0

    # Create the next log file name
    next_log_file = f'log_{max_number + 1}.txt'
    return next_log_file

def read_txt_files_ordered(folder_path):
    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Create a list of tuples (file_name, creation_time)
    file_times = []
    for file in txt_files:
        file_path = os.path.join(folder_path, file)
        creation_time = os.path.getctime(file_path)
        file_times.append((file, creation_time))
    
    # Sort the list based on creation time
    file_times.sort(key=lambda x: x[1])
    
    # Read and store the content of each file in order
    ordered_contents = []
    for file, _ in file_times:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            content = f.readlines()
        ordered_contents.extend(content)
    
    return ordered_contents

# --------------------------------------------------------------------------------------------------------------------------------------------------------------