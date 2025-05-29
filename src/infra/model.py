import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 32×32
            nn.ReLU(),
            nn.Conv2d(32,64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.Conv2d(64,128,4, stride=2, padding=1),  # 8×8
            nn.ReLU(),
            nn.Conv2d(128,256,4,stride=2,padding=1),   # 4×4
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu  = nn.Linear(256*4*4, z_dim)
        self.fc_logv= nn.Linear(256*4*4, z_dim)

    def forward(self, x):
        h = self.conv(x)
        return self.fc_mu(h), self.fc_logv(h)

class Decoder(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*4*4)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (256,4,4)),
            nn.ConvTranspose2d(256,128,4,2,1),  # 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(128,64, 4,2,1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32,4,2,1),   # 32×32
            nn.ReLU(),
            nn.ConvTranspose2d(32,  3,4,2,1),   # 64×64
            nn.Tanh(),                          # output in [–1,1]
        )

    def forward(self, z):
        h = self.fc(z)
        return self.deconv(h)

class FrameAE(nn.Module):
    def __init__(self, z_dim=64, beta=1e-3):
        super().__init__()
        self.enc  = Encoder(z_dim)
        self.dec  = Decoder(z_dim)
        self.beta = beta

    def reparam(self, mu, logv):
        std = (0.5*logv).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logv = self.enc(x)
        z        = self.reparam(mu, logv)
        recon    = self.dec(z)
        # recon loss
        mse = F.mse_loss(recon, x, reduction="mean")
        # KL divergence to N(0,I)
        kl  = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        return recon, mu, logv, mse + self.beta*kl, mse, kl