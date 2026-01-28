import torch
from torch import nn
import os

def generate_and_save_random_weights(input_dim, output_dim, filename='model_weights.pth'):
    """
    Generates random weights for a Linear layer with the given input and output dimensions,
    and saves them in .pth format so they can be loaded with torch.load and used with
    m.load_state_dict(saved_weights).
    """
    # Create a Linear layer
    model = nn.Linear(input_dim, output_dim)
    # Get the state dict (contains 'weight' and 'bias')
    state_dict = model.state_dict()
    # Save the state dict to the specified file
    torch.save(state_dict, filename)
    print(f"Random weights saved to {filename}")


def get_weights_path(filename='model_weights.pth'):
    """
    Returns the absolute path to the weights file in the current directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

if __name__ == "__main__":
    generate_and_save_random_weights(128, 128, get_weights_path('model_weights.pth'))