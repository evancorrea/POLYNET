import torch
import torch.nn as nn
import torch.nn.functional as F

class POLYNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size = 8, padding = 'same')
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size = 6, padding='same')
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.gmp(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).squeeze(-1)