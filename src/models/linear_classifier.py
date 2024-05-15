import torch
import torch.nn as nn

class LinearClassifier(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.fc = nn.Linear(512 * 2, configs.num_classes)
    
    def forward(self, ecg, text):
        embeddings = torch.cat((ecg, text), dim=1)
        x = self.fc(embeddings)
        return x
