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

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 60*60)
        self.unflatten = nn.Unflatten(1, (60, 60))

    def forward(self, z):
        h = self.fc(z)
        h = self.unflatten(h)
        return h

class FrameAE(nn.Module):
    def __init__(self, coord_dim=32):
        super().__init__()
        self.enc  = Encoder()
        self.dec = Decoder()
        
        self.downsample_and_grayscale = T.Compose([
                T.Resize((60,60)),
                T.Grayscale(num_output_channels=1),
        ])

    def forward(self, x):
        coords = self.enc(x)

        recon = self.dec(coords)
        recon_truth = self.downsample_and_grayscale(x).squeeze(1)

        recon_loss = F.mse_loss(recon, recon_truth, reduction="mean")
        
        return recon, recon_truth, recon_loss
        