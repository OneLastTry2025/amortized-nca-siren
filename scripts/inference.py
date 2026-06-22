#!/usr/bin/env python3
"""
Inference script for Amortized NCA with SIREN Decoder.

Loads a trained checkpoint and generates images at arbitrary resolution
in a single forward pass (<10ms on laptop GPU).

Usage:
    python scripts/inference.py --checkpoint checkpoints/best_model.pth --target circle --resolution 512 --output output.png
    python scripts/inference.py --checkpoint checkpoints/best_model.pth --target path/to/target.png --resolution 512 --output output.png
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image

from amortized_nca_siren.models.amortized_nca import AmortizedNCA


def load_model(checkpoint_path: str, device: str = "cuda") -> AmortizedNCA:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = AmortizedNCA(
        latent_dim=config['latent_dim'],
        grid_size=64,
        siren_hidden=config['siren_hidden'],
        siren_layers=config['siren_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
    print(f"   Config: latent_dim={config['latent_dim']}, siren_hidden={config['siren_hidden']}, siren_layers={config['siren_layers']}")
    
    return model


def create_circle_target(size: int = 64, device: str = "cuda") -> torch.Tensor:
    """Create a simple circle target (RGBA)."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing='ij'
    )
    r = torch.sqrt(x**2 + y**2)
    
    # RGBA: circle in red channel, alpha = 1 inside circle
    target = torch.zeros(4, size, size, device=device)
    target[0] = (r < 0.5).float()  # Red channel
    target[3] = (r < 0.5).float()  # Alpha channel
    
    return target.unsqueeze(0)  # [1, 4, H, W]


def load_target_image(path: str, device: str = "cuda") -> torch.Tensor:
    """Load target image from file."""
    img = Image.open(path).convert('RGBA')
    img = img.resize((64, 64), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    target = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return target


def generate_from_target(target: torch.Tensor, model: AmortizedNCA, 
                         resolution: int = 512, device: str = "cuda") -> np.ndarray:
    """
    Generate image at target resolution from input target.
    
    Args:
        target: [1, 4, H, W] - input target image
        model: AmortizedNCA model
        resolution: output resolution
        device: compute device
        
    Returns:
        output: [resolution, resolution, 4] numpy array (0-1 range)
    """
    target = target.to(device)
    
    with torch.no_grad():
        # Single forward pass - encode + decode at target resolution
        generated = model.generate(target, resolution=resolution)
    
    # Convert to numpy [H, W, 4]
    output = generated[0].permute(1, 2, 0).cpu().numpy()
    output = np.clip(output, 0, 1)
    
    return output


def save_image(array: np.ndarray, path: str):
    """Save numpy array as PNG."""
    img = Image.fromarray((array * 255).astype(np.uint8))
    img.save(path)
    print(f"✅ Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Amortized NCA Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--target", required=True, help="Target: 'circle' or path to image file")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"🔧 Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load or create target
    if args.target == "circle":
        target = create_circle_target(64, device)
        print("🎯 Created circle target (64x64)")
    else:
        target = load_target_image(args.target, device)
        print(f"🎯 Loaded target from {args.target}")
    
    # Generate
    import time
    start = time.time()
    output = generate_from_target(target, model, args.resolution, device)
    elapsed = (time.time() - start) * 1000
    
    print(f"⚡ Generated {args.resolution}x{args.resolution} in {elapsed:.1f} ms")
    
    # Save
    save_image(output, args.output)
    
    # Also save a small version for quick viewing
    small = Image.fromarray((output * 255).astype(np.uint8)).resize((256, 256), Image.LANCZOS)
    small.save(args.output.replace('.png', '_256.png'))
    print(f"✅ Also saved 256x256 preview")


if __name__ == "__main__":
    main()