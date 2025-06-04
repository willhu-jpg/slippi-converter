import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """3×3 → BN → ReLU → 3×3 → BN with a skip connection"""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.need_proj = stride != 1 or in_ch != out_ch
        self.proj = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if self.need_proj else nn.Identity()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.proj(x)
        return self.relu(out)

class BigEncoder(nn.Module):
    """
    Input: 3×64×64
    Output: z_dim vector (default 128)
    Depth: 14 conv layers with residual links
    """
    def __init__(self, z_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # Stem (64×64 → 32×32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),     # 32×32
        )

        # Residual stages
        # (channels, blocks, first_stride)
        cfg = [
            (64,  2, 1),   # 32×32
            (128, 2, 2),   # 16×16
            (256, 2, 2),   # 8×8
            (512, 2, 2),   # 4×4
        ]
        stages = []
        in_ch = 64
        for out_ch, n_blk, first_stride in cfg:
            for i in range(n_blk):
                stride = first_stride if i == 0 else 1
                stages.append(
                    ResidualBlock(in_ch, out_ch, stride=stride, dropout=dropout)
                )
                in_ch = out_ch
        self.backbone = nn.Sequential(*stages)

        # Global average-pool (4×4 → 1×1), then FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B, 512, 1, 1)
            nn.Flatten(),                  # (B, 512)
            nn.Linear(512, z_dim),
        )

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        z = self.head(x)
        return z