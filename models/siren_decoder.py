"""
SIREN (Sinusoidal Representation Networks) Neural Field Decoder
for resolution-invariant NCA decoding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SineLayer(nn.Module):
    """Single SIREN layer with sine activation."""
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, is_first: bool = False, 
                 omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Uniform initialization for first layer
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # SIREN initialization
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ModulatedSineLayer(nn.Module):
    """SIREN layer with FiLM-style modulation from latent vector."""
    
    def __init__(self, in_features: int, out_features: int, 
                 latent_dim: int, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.gamma = nn.Linear(latent_dim, out_features)
        self.beta = nn.Linear(latent_dim, out_features)
        
        # Initialize
        with torch.no_grad():
            self.linear.weight.uniform_(
                -np.sqrt(6 / in_features) / omega_0,
                np.sqrt(6 / in_features) / omega_0
            )
            self.gamma.weight.zero_()
            self.beta.weight.zero_()
            self.gamma.bias.data.fill_(1.0)
            self.beta.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_features]
            z: [B, N, latent_dim]
        """
        # Linear transform
        x = self.linear(x)
        
        # FiLM modulation
        gamma = self.gamma(z)  # [B, N, out_features]
        beta = self.beta(z)
        
        x = gamma * x + beta
        
        return torch.sin(self.omega_0 * x)


class SIRENDecoder(nn.Module):
    """
    SIREN-based Neural Field Decoder.
    
    Takes a latent vector z and spatial coordinates (x, y) and outputs
    the pixel value at that coordinate. This enables resolution-inference at any resolution.
    """
    
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, 
                 num_layers: int = 5, output_channels: int = 4,
                 omega_0: float = 30.0, omega_0_hidden: float = 30.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        
        # Input: (x, y) coordinates + latent vector z
        # We concatenate z to the coordinate input at each layer (modulated SIREN)
        # Or we can use FiLM-style modulation
        
        # First layer: 2D coordinates -> hidden_dim
        self.first_layer = SineLayer(2, hidden_dim, is_first=True, omega_0=omega_0)
        
        # Hidden layers with latent modulation
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            self.hidden_layers.append(
                ModulatedSineLayer(hidden_dim, hidden_dim, latent_dim, omega_0=omega_0_hidden)
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_channels)
        # Initialize output layer
        with torch.no_grad():
            self.output_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0_hidden,
                np.sqrt(6 / hidden_dim) / omega_0_hidden
            )
    
    def forward(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, N, 2] or [N, 2] - spatial coordinates (x, y) in [-1, 1]
            z: [B, latent_dim] - latent vector
            
        Returns:
            output: [B, N, output_channels] - pixel values at coordinates
        """
        # Handle both batched and unbatched coords
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # [1, N, 2]
        
        B, N, _ = coords.shape
        
        # Expand z to match coords
        z_expanded = z.unsqueeze(1).expand(-1, N, -1)  # [B, N, latent_dim]
        
        # First layer
        x = self.first_layer(coords)  # [B, N, hidden_dim]
        
        # Hidden layers with modulation
        for layer in self.hidden_layers:
            x = layer(x, z_expanded)
        
        # Output
        output = self.output_layer(x)  # [B, N, output_channels]
        
        return output
    
    def decode_image(self, z: torch.Tensor, resolution: int = 256) -> torch.Tensor:
        """
        Decode latent to full image at given resolution.
        
        Args:
            z: [B, latent_dim]
            resolution: output image size (resolution x resolution)
            
        Returns:
            image: [B, output_channels, resolution, resolution]
        """
        B = z.shape[0]
        device = z.device
        
        # Create coordinate grid
        coords = torch.linspace(-1, 1, resolution, device=device)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        coords_grid = torch.stack([x, y], dim=-1).reshape(1, -1, 2)  # [1, H*W, 2]
        coords_grid = coords_grid.expand(B, -1, -1)  # [B, H*W, 2]
        
        # Decode
        with torch.no_grad():
            output = self.forward(coords_grid, z)  # [B, H*W, C]
        
        # Reshape to image
        image = output.reshape(B, resolution, resolution, self.output_channels)
        image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return image