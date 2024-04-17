import torch
from torch import nn

class TextLinearProjectionHead(nn.Module):
    """
    A linear projection head for text data.
    """
    def __init__(self, input_dim : int, output_dim : int = 512, temperature = 0.07):
        super().__init__()
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x / self.temperature)

class EcgLinearProjectionHead(nn.Module):
    """
    A linear projection head for ECG data.
    """
    def __init__(self, input_dim : int, output_dim : int = 512, temperature = 0.07):
        super().__init__()
        self.temperature = temperature
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x / self.temperature)