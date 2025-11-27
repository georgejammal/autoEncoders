import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ffhq import build_dataloaders
from src.utils import set_seed

# Hardcoded configurations for known runs
RUN_CONFIGS = {
    "run_D1_deep_resnet_vae": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetVAE",
        "kwargs": {"latent_dim": 256},
        "is_vae": True
    },
    "run_D2_deep_resnet_vae_lpips": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetVAE",
        "kwargs": {"latent_dim": 256},
        "is_vae": True
    },
    "run_D3_attention_ae": {
        "module": "src.models.vae_attention",
        "class": "DeepResNetAttentionAE",
        "kwargs": {"latent_dim": 256},
        "is_vae": False
    },
    "run_D4_attention_vae_lpips": {
        "module": "src.models.vae_attention",
        "class": "DeepResNetAttentionVAE",
        "kwargs": {"latent_dim": 256},
        "is_vae": True
    },
    "run_B2_resnet_lpips_w000": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetAutoencoder",
        "kwargs": {"latent_dim": 256},
        "is_vae": False
    },
    "run_B2_resnet_lpips_w005": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetAutoencoder",
        "kwargs": {"latent_dim": 256},
        "is_vae": False
    },
    "run_B2_resnet_lpips_w0075": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetAutoencoder",
        "kwargs": {"latent_dim": 256},
        "is_vae": False
    },
    "run_B2_resnet_lpips_w010": {
        "module": "src.models.deep_resnet_autoencoder",
        "class": "DeepResNetAutoencoder",
        "kwargs": {"latent_dim": 256},
        "is_vae": False
    },
    "run_C1_vae_beta_small": {
        "module": "src.models.vae",
        "class": "VAE",
        "kwargs": {"latent_dim": 256, "base_channels": 32},
        "is_vae": True
    },
    "run_C2_vae_beta_medium": {
        "module": "src.models.vae",
        "class": "VAE",
        "kwargs": {"latent_dim": 256, "base_channels": 32},
        "is_vae": True
    },
    "run_C3_vae_beta_large": {
        "module": "src.models.vae",
        "class": "VAE",
        "kwargs": {"latent_dim": 256, "base_channels": 32},
        "is_vae": True
    },
    "run_C4_vae_wide": {
        "module": "src.models.vae",
        "class": "VAE",
        "kwargs": {"latent_dim": 256, "base_channels": 64},
        "is_vae": True
    },
    "run_C5_vae_lpips": {
        "module": "src.models.vae",
        "class": "VAE",
        "kwargs": {"latent_dim": 256, "base_channels": 32},
        "is_vae": True
    }
}

def build_model(module_path: str, class_name: str, model_kwargs: dict) -> nn.Module:
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**model_kwargs)

def main():
    parser = argparse.ArgumentParser(description="Generate reconstructions from trained models.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to load.")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file name (default: best.pt).")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to save.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save images.")
    parser.add_argument("--mode", type=str, choices=["random", "best", "worst"], default="random", help="Selection mode.")
    
    args = parser.parse_args()
    
    # Resolve config
    if args.run_id in RUN_CONFIGS:
        config = RUN_CONFIGS[args.run_id]
    else:
        print(f"Unknown run ID: {args.run_id}. Please add it to RUN_CONFIGS in the script.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model {config['class']} for {args.run_id}...")
    model = build_model(config["module"], config["class"], config["kwargs"]).to(device)
    
    checkpoint_path = PROJECT_ROOT / "experiments" / "runs" / args.run_id / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Data
    print("Loading validation data...")
    _, val_loader = build_dataloaders(
        root=Path("/home/ML_courses/03683533_2025/dataset"),
        test_split=1000,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=256
    )

    # Output Directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "report_assets" / "reconstructions" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to {out_dir}")

    results = []

    print("Running inference...")
    with torch.no_grad():
        for images, _ in tqdm(val_loader):
            images = images.to(device)
            
            if config["is_vae"]:
                recon, _, _ = model(images)
            else:
                recon = model(images)
            
            # Calculate loss per image for sorting
            if args.mode != "random":
                losses = torch.mean(torch.abs(images - recon), dim=[1, 2, 3])
                for i in range(images.size(0)):
                    results.append({
                        "original": images[i].cpu(),
                        "recon": recon[i].cpu(),
                        "loss": losses[i].item()
                    })
            else:
                # Random mode: just save as we go until we have enough
                for i in range(images.size(0)):
                    results.append({
                        "original": images[i].cpu(),
                        "recon": recon[i].cpu(),
                        "loss": 0.0 # Dummy
                    })
                if len(results) >= args.num_images:
                    break

    # Sort if needed
    if args.mode == "best":
        results.sort(key=lambda x: x["loss"])
    elif args.mode == "worst":
        results.sort(key=lambda x: x["loss"], reverse=True)
    
    # Save images
    print(f"Saving top {args.num_images} images...")
    for i, item in enumerate(results[:args.num_images]):
        # Denormalize
        orig = (item["original"].clamp(-1, 1) + 1) * 0.5
        rec = (item["recon"].clamp(-1, 1) + 1) * 0.5
        
        # Combine side-by-side
        combined = torch.cat([orig, rec], dim=2) # Concatenate along width
        
        save_path = out_dir / f"{args.mode}_{i:03d}_loss_{item['loss']:.4f}.png"
        save_image(combined, save_path)

    print("Done!")

if __name__ == "__main__":
    main()
