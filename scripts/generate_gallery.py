#!/usr/bin/env python3
"""
Generate asset gallery: SSIM curve plot and sample images at multiple resolutions.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amortized_nca_siren.models.amortized_nca import AmortizedNCA


def load_model(checkpoint_path: str, device: str = "cpu") -> AmortizedNCA:
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
    
    return model


def create_circle_target(size: int = 64, device: str = "cpu") -> torch.Tensor:
    """Create a simple circle target (RGBA)."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing='ij'
    )
    r = torch.sqrt(x**2 + y**2)
    
    target = torch.zeros(4, size, size, device=device)
    target[0] = (r < 0.5).float()  # Red channel
    target[3] = (r < 0.5).float()  # Alpha channel
    
    return target.unsqueeze(0)


def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute SSIM between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, 1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, 1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, 1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def evaluate_resolutions(model: AmortizedNCA, target: torch.Tensor, 
                         resolutions: list, device: str = "cpu") -> dict:
    """Evaluate model at multiple resolutions."""
    results = {}
    
    for res in resolutions:
        with torch.no_grad():
            generated = model.generate(target, resolution=res)
            
            # Resize target to match for comparison
            target_resized = torch.nn.functional.interpolate(
                target, size=(res, res), mode='bilinear', align_corners=False
            )
            
            mse = torch.nn.functional.mse_loss(generated, target_resized).item()
            ssim = compute_ssim(generated, target_resized)
            
            results[res] = {'mse': mse, 'ssim': ssim}
            print(f"  {res}x{res}: MSE={mse:.6f}, SSIM={ssim:.4f}")
    
    return results


def plot_ssim_curve(results: dict, output_path: str):
    """Plot SSIM vs Resolution curve."""
    resolutions = sorted(results.keys())
    ssims = [results[r]['ssim'] for r in resolutions]
    mses = [results[r]['mse'] for r in resolutions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # SSIM curve
    ax1.plot(resolutions, ssims, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('SSIM', fontsize=12)
    ax1.set_title('SSIM vs Resolution (Circle Target)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.85, 0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(resolutions)
    
    # Add value labels
    for r, s in zip(resolutions, ssims):
        ax1.annotate(f'{s:.4f}', (r, s), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10)
    
    # MSE curve
    ax2.plot(resolutions, mses, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Resolution', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE vs Resolution (Circle Target)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(resolutions)
    
    for r, m in zip(resolutions, mses):
        ax2.annotate(f'{m:.6f}', (r, m), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ SSIM curve saved to {output_path}")


def save_sample_images(model: AmortizedNCA, target: torch.Tensor, 
                       resolutions: list, output_dir: str, device: str = "cpu"):
    """Save sample images at each resolution."""
    from PIL import Image
    
    for res in resolutions:
        with torch.no_grad():
            generated = model.generate(target, resolution=res)
        
        # Convert to numpy [H, W, 4]
        output = generated[0].permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)
        
        # Save
        path = os.path.join(output_dir, f"sample_{res}.png")
        img = Image.fromarray((output * 255).astype(np.uint8))
        img.save(path)
        print(f"✅ Saved {res}x{res} sample to {path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate asset gallery")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="assets", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device: cuda or cpu")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"🔧 Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create target
    target = create_circle_target(64, device)
    
    # Resolutions to evaluate
    resolutions = [64, 128, 256, 512]
    
    # Evaluate
    print("\n📊 Evaluating at multiple resolutions...")
    results = evaluate_resolutions(model, target, resolutions, device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Plot SSIM curve
    plot_ssim_curve(results, os.path.join(args.output, "ssim_curve.png"))
    
    # Save sample images
    print("\n🖼️  Saving sample images...")
    save_sample_images(model, target, resolutions, args.output, device)
    
    # Print summary table
    print("\n📋 Summary Table:")
    print(f"{'Resolution':<12} {'SSIM':<10} {'MSE':<12}")
    print("-" * 34)
    for res in resolutions:
        print(f"{res}x{res:<6} {results[res]['ssim']:<10.4f} {results[res]['mse']:<12.6f}")
    
    print(f"\n✅ Gallery generated in {args.output}/")


if __name__ == "__main__":
    main()