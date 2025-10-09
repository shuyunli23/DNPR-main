import torch
import os
import numpy as np
from torch import nn
import sys


def print_memory_allocated() -> None:
    """
    Print the currently allocated GPU memory (in bytes).
    """
    allocated_memory = torch.cuda.memory_allocated()
    print(f"Currently allocated GPU memory: {allocated_memory / (1024 ** 2):.2f} MB")


def print_memory_reserved() -> None:
    """
    Print the currently reserved GPU memory (in bytes).
    """
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Currently reserved GPU memory: {reserved_memory / (1024 ** 2):.2f} MB")


def clear_cuda_cache() -> None:
    """
    Clear the unused memory from the GPU cache.
    This helps reduce fragmentation and free up memory for other operations.
    """
    torch.cuda.empty_cache()
    print("Cleared unused GPU memory from the cache.")


def save_tensor_or_npy_to_npy(input_data, filename):
    """
    Save the given Tensor or .npy file to a .npy file, creating the directory if it doesn't exist.

    Parameters:
    input_data -- The Tensor (PyTorch or TensorFlow) or a NumPy array or .npy file to be saved
    filename -- The name of the file to save (including path)
    """
    # Check if input_data is a Tensor and convert to NumPy array
    if hasattr(input_data, 'numpy'):  # TensorFlow or PyTorch
        numpy_array = input_data.numpy()
    elif isinstance(input_data, np.ndarray):  # Already a NumPy array
        numpy_array = input_data
    elif isinstance(input_data, str) and input_data.endswith('.npy'):  # If input is a .npy file
        numpy_array = np.load(input_data)  # Load the .npy file
    else:
        raise ValueError("Input must be a Tensor, NumPy array, or a .npy file path.")

    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save to .npy file
    np.save(filename, numpy_array)
    print(f"Data has been saved as {filename}")


def split_scores_by_mask(scores, mask):
    """
    Split the scores tensor based on a binary mask.

    Parameters:
    scores -- A tensor containing the scores.
    mask -- A binary tensor where 0 and 1 indicate the positions.

    Returns:
    scores_0 -- Flattened scores at positions where mask is 0.
    scores_1 -- Flattened scores at positions where mask is 1.
    """
    # Ensure the scores and mask are the same shape
    assert scores.shape == mask.shape, "Scores and mask must have the same shape"

    # Move tensors to CPU if they are on GPU
    if scores.is_cuda:
        scores = scores.cpu()
    if mask.is_cuda:
        mask = mask.cpu()

    # Use boolean indexing to separate scores
    scores_0 = scores[mask == 0].view(-1)
    scores_1 = scores[mask == 1].view(-1)

    return scores_0, scores_1


def create_directory(*dirs):
    """
    Create directories if they do not already exist.

    Args:
        *dirs (str): One or more directory paths to create.

    Raises:
        OSError: If any directory cannot be created due to system-level errors.
    """
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)  # `exist_ok=True` avoids error if directory already exists
        except OSError as e:
            print(f"Error creating directory {dir_path}: {e}")


def get_memory_usage(obj, device=None):
    """
    Calculate the memory usage of a given object (in MB).

    Parameters:
    obj: The object whose memory usage is to be calculated. It can be a tensor, model, list, set, dict, or custom class.
    device: The specific GPU device (e.g., "cuda:0") to check memory usage on. If None, checks all devices.

    Returns:
    float: Memory usage of the object (in MB).
    """
    total_memory = 0

    # Function to handle tensor or model memory calculation
    def calculate_memory(item):
        nonlocal total_memory
        if isinstance(item, torch.Tensor):
            if device is None or item.device == device:
                total_memory += item.element_size() * item.nelement()
        elif isinstance(item, nn.Module):
            for param in item.parameters():
                if device is None or param.device == device:
                    total_memory += param.element_size() * param.nelement()
            for buffer in item.buffers():
                if device is None or buffer.device == device:
                    total_memory += buffer.element_size() * buffer.nelement()
        elif isinstance(item, (list, set, dict)):
            # Recursively process nested structures
            if isinstance(item, list):
                for sub_item in item:
                    calculate_memory(sub_item)
            elif isinstance(item, set):
                for sub_item in item:
                    calculate_memory(sub_item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    calculate_memory(key)
                    calculate_memory(value)
        # Check if the item is a custom class
        elif hasattr(item, '__dict__'):
            for attr_name, attr_value in item.__dict__.items():
                calculate_memory(attr_value)
        # For regular Python objects, use sys.getsizeof()
        elif device is None:
            total_memory += sys.getsizeof(item)

    # Start calculating memory for the initial object
    calculate_memory(obj)

    # Convert to MB
    total_memory_mb = total_memory / (1024 ** 2)  # Convert to MB
    return total_memory_mb
