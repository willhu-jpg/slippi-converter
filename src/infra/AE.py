import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from infra.SpatialSoftMax import SpatialSoftArgmax

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 240x240
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=0),  # 117x117
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, stride=1, padding=0),  # 113x113
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, 5, stride=1, padding=0),  # 109x109
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.spatial_softmax = SpatialSoftArgmax(16, 109, 109)
        self.flatten = nn.Flatten()

    def forward(self, x):
        h = self.conv(x)
        h = self.spatial_softmax(h)
        h = self.flatten(h)
        return h

class FrameAE(nn.Module):
    def __init__(self, coord_dim=32):
        super().__init__()
        self.enc  = Encoder()

        # Predict positions directly from spatial softmax coords
        self.position_head = nn.Linear(coord_dim, 4)  # e.g., p1_x,p1_y,p2_x,p2_y

        # Additional head for percentages
        self.percent_head = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # player 1 percent, player 2 percent
        )

        # Additional head for facing
        self.facing_head = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # player 1 facing, player 2 facing
        )

        # Additional head for action
        self.action_head = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # player 1 action, player 2 action
        )

    def forward(self, x):
        coords = self.enc(x)
        positions = self.position_head(coords)
        percentages = self.percent_head(coords)
        facings = self.facing_head(coords)
        actions = self.action_head(coords)
        
        return positions, percentages, facings, actions
        