import torch
from torch import nn
import torch.nn.functional as F

class SpatialSoftArgmax(nn.Module):
    """
    Given input feature‐maps of shape (B, C, H, W), this module:
      1. Applies a per‐channel spatial softmax with learned "temperature" α
         over the H×W activations.
      2. Computes expected 2D coordinates (x, y) for each channel:
         f_c = ( Σ_i Σ_j i * softmax_ij , Σ_i Σ_j j * softmax_ij ).
      3. Returns a tensor of shape (B, 2*C), where for each sample and channel
         we produce the pair (x_c, y_c).
    """
    def __init__(self, num_channels: int, height: int, width: int):
        super().__init__()
        self.num_channels = num_channels
        self.height = height
        self.width = width

        # Learnable "temperature" parameter α (initialized to 1.0 by default)
        self.log_alpha = nn.Parameter(torch.zeros(()))  # stores log(α)

        # Precompute coordinate grids of shape (H, W)
        # i_grid holds row‐indices [0, 1, …, H−1], j_grid holds col‐indices [0, 1, …, W−1].
        # We register them as buffers so they move with `.to(device)` calls.
        i_range = torch.linspace(0, height - 1, steps=height)
        j_range = torch.linspace(0, width - 1,  steps=width)
        # create (H, W) grids
        i_grid, j_grid = torch.meshgrid(i_range, j_range, indexing='ij')
        # Clone the tensors to avoid memory sharing issues when loading state dict
        self.register_buffer('i_grid', i_grid.clone())  # shape (H, W)
        self.register_buffer('j_grid', j_grid.clone())  # shape (H, W)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """
        feature_maps: (B, C, H, W)
        returns:       (B, 2*C)  concatenated [x_1, y_1, x_2, y_2, …, x_C, y_C]
        """
        B, C, H, W = feature_maps.shape
        assert C == self.num_channels and H == self.height and W == self.width

        # Compute temperature α = exp(log_alpha) to ensure positivity
        alpha = torch.exp(self.log_alpha)

        # Reshape to (B * C, H * W) so we can do a single softmax over each channel map
        fmap_flat = feature_maps.view(B * C, H * W)  # (B*C, H*W)
        # Divide by temperature, then apply softmax over the spatial dim:
        softmax_flat = F.softmax(fmap_flat / alpha, dim=-1)  # (B*C, H*W)

        # Reshape back to spatial maps (B, C, H, W)
        softmax_maps = softmax_flat.view(B, C, H, W)

        # Now compute expected coordinates per channel:
        #   x_coord[c] = Σ_i Σ_j (j * P_ij),
        #   y_coord[c] = Σ_i Σ_j (i * P_ij)
        #
        # We have i_grid, j_grid each of shape (H, W). We'll multiply and sum.

        # Expand grids to (1, 1, H, W) so broadcasting works against (B, C, H, W)
        i_grid = self.i_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        j_grid = self.j_grid.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Multiply probability map by coordinate grid and sum over H, W:
        #   sum_i sum_j i * P_ij  → shape (B, C)
        #   sum_i sum_j j * P_ij  → shape (B, C)
        y_coords = torch.sum(softmax_maps * i_grid, dim=(-2, -1))  # (B, C)
        x_coords = torch.sum(softmax_maps * j_grid, dim=(-2, -1))  # (B, C)

        # Concatenate x‐ and y‐coordinates for each channel:
        # final shape → (B, 2*C)
        coords = torch.cat([x_coords, y_coords], dim=1)  # (B, 2C)
        return coords