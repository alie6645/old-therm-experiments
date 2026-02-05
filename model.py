import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32*3, 32*32*3),
            nn.ReLU(),
            nn.Linear(32*32*3, 32*32*3),
            nn.ReLU(),
            nn.Linear(32*32*3, 32*32*1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits