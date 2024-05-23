import torch
import torch.nn as nn
import torch.nn.init as init
import math

class LinearClassifier(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.name = "LinearClassifier"
        self.fc = nn.Linear(512 * 2, configs.num_classes)
        self._initialize_weights()
    
    def forward(self, ecg, text):
        embeddings = torch.cat((ecg, text), dim=1)
        x = self.fc(embeddings)
        return x
    
    def _initialize_weights(self):
        init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.fc.bias, -bound, bound)
