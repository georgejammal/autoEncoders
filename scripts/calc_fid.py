import argparse
import json
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ffhq import build_dataloaders
from src.utils import set_seed

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Error: torch-fidelity is not installed.")
    print("Please install it using: pip install torch-fidelity")
    sys.exit(1)


def build_model(module_path: str, class_name: str, model_kwargs: dict) -> nn.Module:
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**model_kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID for trained models")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to evaluate")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file name (e.g., best.pt, last.pt)")
    parser.add_argument("--mode", type=str, choices=["generation", "reconstruction"], default="generation", help="FID mode")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples for FID calculation")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation/inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ML_courses/03683533_2025/dataset"), help="Path to dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. Setup directories
    run_dir = PROJECT_ROOT / "experiments" / "runs" / args.run_id
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        sys.exit(1)

    # Load run args to get model config
    # We assume the user passes the same args or we can try to parse them from a log/config if available.
    # For simplicity, we will require the user to pass the model config args again or 
    # we can try to infer them. 
    # Actually, the best way is to look at the launch script or just hardcode for this task since we know the models.
    # But to be generic, let's look for a config file? The training script didn't save a config.json.
    # We'll have to rely on the user providing the model details or hardcode a mapping based on run_id.
    
    # Mapping based on run_id naming convention from Part D
    if "resnet_vae" in args.run_id:
        model_module = "src.models.deep_resnet_autoencoder"
        model_class = "DeepResNetVAE"
        model_kwargs = {"latent_dim": 256}
        is_vae = True
    elif "attention_ae" in args.run_id:
        model_module = "src.models.vae_attention"
        model_class = "DeepResNetAttentionAE"
        model_kwargs = {"latent_dim": 256}
        is_vae = False
    elif "attention_vae" in args.run_id:
        model_module = "src.models.vae_attention"
        model_class = "DeepResNetAttentionVAE"
        model_kwargs = {"latent_dim": 256}
        is_vae = True
    else:
        print(f"Error: Could not infer model config from run_id {args.run_id}. Please extend the script.")
        sys.exit(1)

    print(f"Loading {model_class} from {run_dir}...")

    # 2. Load Model
    model = build_model(model_module, model_class, model_kwargs).to(device)
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Prepare Data Folders for FID
    # We need two folders: 'real' and 'fake'
    fid_dir = run_dir / "fid_results"
    fid_dir.mkdir(exist_ok=True)
    
    real_dir = fid_dir / "real"
    fake_dir = fid_dir / "fake"
    
    # Clean up previous runs
    if real_dir.exists(): shutil.rmtree(real_dir)
    if fake_dir.exists(): shutil.rmtree(fake_dir)
    real_dir.mkdir()
    fake_dir.mkdir()

    print(f"Generating {args.num_samples} samples in {args.mode} mode...")

    # 4. Generate/Extract Images
    
    # We need the validation loader to get real images (and for reconstruction mode)
    # Note: We only need 'args.num_samples' images.
    # The training script used val_split=1000. We should probably use that or more.
    # If num_samples > val_split, we might need to use train set too, but for FID 
    # we usually compare against the validation set distribution.
    # Let's use the first N images from the dataset (which is the validation split logic).
    
    _, val_loader = build_dataloaders(
        root=args.dataset_root,
        test_split=args.num_samples, # Get enough samples
        batch_size=args.batch_size,
        num_workers=4,
        image_size=256
    )

    count = 0
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(val_loader, desc="Processing images")):
            if count >= args.num_samples:
                break
            
            images = images.to(device)
            bs = images.size(0)
            
            # Save Real Images
            # Denormalize: [-1, 1] -> [0, 1]
            real_imgs = (images.clamp(-1, 1) + 1.0) * 0.5
            
            for i in range(bs):
                if count + i >= args.num_samples:
                    break
                save_image(real_imgs[i], real_dir / f"{count+i:05d}.png")

            # Generate Fake Images
            if args.mode == "generation":
                if not is_vae:
                    print("Error: Generation mode not supported for AE (no sampling). Use --mode reconstruction.")
                    sys.exit(1)
                
                # Sample from prior N(0, I)
                z = torch.randn(bs, model_kwargs["latent_dim"]).to(device)
                recon = model.decode(z)
                
            elif args.mode == "reconstruction":
                if is_vae:
                    recon, _, _ = model(images)
                else:
                    recon = model(images)
            
            # Save Fake Images
            fake_imgs = (recon.clamp(-1, 1) + 1.0) * 0.5
            for i in range(bs):
                if count + i >= args.num_samples:
                    break
                save_image(fake_imgs[i], fake_dir / f"{count+i:05d}.png")
            
            count += bs

    print("Calculating FID...")
    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        cuda=True,
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
    )
    
    print(f"FID Score ({args.mode}): {metrics['frechet_inception_distance']:.4f}")
    
    # Save result
    result_file = fid_dir / "fid_score.txt"
    with open(result_file, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Num Samples: {args.num_samples}\n")
        f.write(f"FID: {metrics['frechet_inception_distance']:.4f}\n")
    
    print(f"Saved results to {result_file}")

if __name__ == "__main__":
    main()
