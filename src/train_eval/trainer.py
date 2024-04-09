"""
This script contains the Trainer class which is used to train and evaluate the model.
"""

import torch

class Trainer:
    """
    Class to train and evaluate torch.nn models.
    """
    def __init__(self, optimizer, criterion):
        self.optimizer = optimizer
        self.criterion = criterion
    def train(model, train_data, device):
        pass