#!/usr/bin/env python3
"""
Generate high-quality loss curve plots using matplotlib for the report.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "report_assets" / "loss_curves"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define all models by part
MODELS = {
    "Part A": [
        "run_A1_skip_convae_depth10",
        "run_A2_skip_convae_bn",
        "run_A3_skip_convae_leaky",
        "run_A4_skip_convae_wide",
    ],
    "Part B": [
        "run_B1_resnet_ae",
        "run_B2_resnet_lpips_w000",
        "run_B2_resnet_lpips_w005",
        "run_B2_resnet_lpips_w0075",
        "run_B2_resnet_lpips_w010",
    ],
    "Part C": [
        "run_C1_vae_beta_small",
        "run_C2_vae_beta_medium",
        "run_C3_vae_beta_large",
        "run_C4_vae_wide",
        "run_C5_vae_lpips",
    ],
    "Part D": [
        "run_D1_deep_resnet_vae",
        "run_D2_deep_resnet_vae_lpips",
        "run_D3_attention_ae",
    ],
}


def load_metrics(run_id):
    """Load metrics.json for a given run."""
    metrics_path = EXPERIMENTS_DIR / run_id / "metrics.json"
    if not metrics_path.exists():
        print(f"  ⚠️  No metrics found for {run_id}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_loss_curves(run_id, metrics, output_path):
    """Create professional loss curves with matplotlib."""
    # Handle different key names
    train_key = 'train_l1' if 'train_l1' in metrics else 'train_loss'
    val_key = 'val_l1' if 'val_l1' in metrics else 'val_loss'
    
    train_loss = metrics.get(train_key, [])
    val_loss = metrics.get(val_key, [])
    
    if not train_loss:
        return
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = np.arange(1, len(train_loss) + 1)
    
    # Plot training and validation loss
    ax.plot(epochs, train_loss, label='Training Loss', linewidth=2.5, 
            color='#2E86AB', marker='o', markersize=4, markevery=max(1, len(epochs)//10))
    
    if val_loss:
        ax.plot(epochs, val_loss, label='Validation Loss', linewidth=2.5, 
                color='#A23B72', marker='s', markersize=4, markevery=max(1, len(epochs)//10))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title(f'{run_id}', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Set integer ticks on x-axis
    ax.set_xticks(np.linspace(1, len(epochs), min(10, len(epochs)), dtype=int))
    
    # Improve tick label size
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_all_plots():
    """Generate plots for all models."""
    print("=" * 60)
    print("Generating High-Quality Loss Curves with Matplotlib")
    print("=" * 60)
    
    total_generated = 0
    
    for part_name, run_ids in MODELS.items():
        print(f"\n{part_name}:")
        print("-" * 40)
        
        for run_id in run_ids:
            metrics = load_metrics(run_id)
            if metrics is None:
                continue
            
            output_path = OUTPUT_DIR / f"{run_id}.png"
            plot_loss_curves(run_id, metrics, output_path)
            
            print(f"  ✓ {run_id} -> {output_path.name}")
            total_generated += 1
    
    print("\n" + "=" * 60)
    print(f"✓ Generated {total_generated} high-quality PNG plots")
    print(f"✓ Saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_plots()
