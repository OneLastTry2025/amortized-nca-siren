"""
Amortized Encoder: maps target image to latent vector z.

This replaces the iterative NCA forward pass with a single forward pass.
"""

import torch
import torch.nn as nn


class AmortizedEncoder(nn.Module):
    """
    Amortized Encoder: maps target image to latent vector z.
    
    This replaces the iterative NCA forward pass with a single forward pass.
    """
    
    def __init__(self, input_channels: int = 4, latent_dim: int = 256,
                 hidden_dim: int = 512, grid_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        # Encoder: CNN that compresses image to latent
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 8 -> 4
            nn.ReLU(),
            nn.Conv2d(512, hidden_dim, 4, 1, 0),  # 4 -> 1
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Mean and logvar for VAE
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [B, C, H, W] - target image
            
        Returns:
            z: [B, latent_dim] - sampled latent
            mu: [B, latent_dim] - mean
            logvar: [B, latent_dim] - log variance
        """
        params = self.encoder(x)  # [B, latent_dim * 2]
        mu, logvar = params.chunk(2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar
    
    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding (for inference)."""
        params = self.encoder(x)
        mu, _ = params.chunk(2, dim=-1)
        return mu