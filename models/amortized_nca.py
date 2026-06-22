"""
Full Amortized NCA: Encoder + SIREN Decoder.

This is the O(1) inference model that replaces iterative NCA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from amortized_nca_siren.models.encoder import AmortizedEncoder
from amortized_nca_siren.models.siren_decoder import SIRENDecoder


class AmortizedNCA(nn.Module):
    """
    Full Amortized NCA: Encoder + SIREN Decoder.
    
    This is the O(1) inference model that replaces iterative NCA.
    """
    
    def __init__(self, latent_dim: int = 256, grid_size: int = 64,
                 siren_hidden: int = 256, siren_layers: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        self.encoder = AmortizedEncoder(
            input_channels=4, 
            latent_dim=latent_dim,
            grid_size=grid_size
        )
        
        self.decoder = SIRENDecoder(
            latent_dim=latent_dim,
            hidden_dim=siren_hidden,
            num_layers=siren_layers,
            output_channels=4
        )
    
    def forward(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            target: [B, 4, H, W] - target image
            
        Returns:
            dict with reconstructed image, latent, mu, logvar
        """
        z, mu, logvar = self.encoder(target)
        
        # Reconstruct at training resolution
        B = target.shape[0]
        H, W = target.shape[2], target.shape[3]
        
        # Create coordinate grid at target resolution
        coords = torch.linspace(-1, 1, W, device=target.device)
        y_coords = torch.linspace(-1, 1, H, device=target.device)
        y, x = torch.meshgrid(y_coords, coords, indexing='ij')
        coords_grid = torch.stack([x, y], dim=-1).reshape(1, -1, 2)
        coords_grid = coords_grid.expand(B, -1, -1)
        
        recon = self.decoder(coords_grid, z)
        recon = recon.reshape(B, H, W, 4).permute(0, 3, 1, 2)
        
        return {
            'recon': recon,
            'z': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def generate(self, target: torch.Tensor, resolution: int = 512) -> torch.Tensor:
        """
        Generate image at arbitrary resolution (inference).
        
        Args:
            target: [B, 4, H, W] - target image (any resolution)
            resolution: output resolution
            
        Returns:
            generated: [B, 4, resolution, resolution]
        """
        z = self.encoder.encode_deterministic(target)
        return self.decoder.decode_image(z, resolution)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


def vae_loss(recon: torch.Tensor, target: torch.Tensor, 
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """VAE loss: reconstruction + KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss
    }