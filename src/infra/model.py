import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim=10, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32×32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(32,64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64,128,4, stride=2, padding=1),  # 8×8
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(128,256,4,stride=2,padding=1),   # 4×4
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.fc  = nn.Linear(256*4*4, z_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h)