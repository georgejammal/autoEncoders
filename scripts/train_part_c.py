import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ffhq import build_dataloaders
from src.models import VAE, PerceptualLoss
from src.utils import ensure_dir, set_seed

# Define VAE experiments (All latent_dims <= 256)
PART_C_RUNS = {
    "run_C1_vae_beta_small": {
        "model_kwargs": {"latent_dim": 256, "base_channels": 32},
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 2e-4,
        "kld_weight": 0.00025,
        "lpips_weight": 0.0,
    },
    "run_C2_vae_beta_medium": {
        "model_kwargs": {"latent_dim": 256, "base_channels": 32},
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 2e-4,
        "kld_weight": 0.001,
        "lpips_weight": 0.0,
    },
    "run_C3_vae_beta_large": {
        "model_kwargs": {"latent_dim": 256, "base_channels": 32},
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 2e-4,
        "kld_weight": 0.01,
        "lpips_weight": 0.0,
    },
    "run_C4_vae_wide": {
        "model_kwargs": {"latent_dim": 256, "base_channels": 64}, # Wider network
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 2e-4,
        "kld_weight": 0.00025,
        "lpips_weight": 0.0,
    },
    "run_C5_vae_lpips": {
        "model_kwargs": {"latent_dim": 256, "base_channels": 32},
        "batch_size": 8,
        "epochs": 20,
        "learning_rate": 2e-4,
        "kld_weight": 0.00025,
        "lpips_weight": 0.1, # Add perceptual loss
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train Part C VAE.")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        choices=sorted(PART_C_RUNS.keys()),
        help="Run ID corresponding to Part C experiment.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ML_courses/03683533_2025/dataset"))
    parser.add_argument("--val-split", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--vis-interval", type=int, default=15)
    return parser.parse_args()

def vae_loss_fn(recon_x, x, mu, logvar, kld_weight=1.0, lpips_loss_fn=None, lpips_weight=0.0):
    """
    Computes VAE loss: Reconstruction + kld_weight * KL Divergence + lpips_weight * LPIPS
    """
    # 1. Reconstruction Loss (L1)
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')
    
    # 2. KL Divergence
    # KLD = -0.5 * sum(1 + log(var) - mu^2 - var)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = kld_loss.mean()
    
    # 3. LPIPS Loss (Optional)
    perc_loss = torch.tensor(0.0, device=x.device)
    if lpips_loss_fn is not None and lpips_weight > 0:
        perc_loss = lpips_loss_fn(recon_x, x)
    
    total_loss = recon_loss + kld_weight * kld_loss + lpips_weight * perc_loss
    return total_loss, recon_loss, kld_loss, perc_loss

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)

def main():
    args = parse_args()
    run_cfg = PART_C_RUNS[args.run_id]
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Data
    train_loader, val_loader = build_dataloaders(
        root=args.dataset_root,
        test_split=args.val_split,
        batch_size=run_cfg["batch_size"],
        num_workers=args.num_workers,
        image_size=256,
    )
    
    # Initialize Model
    model = VAE(**run_cfg["model_kwargs"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg["learning_rate"])
    
    # Initialize LPIPS if needed
    lpips_loss_fn = None
    if run_cfg.get("lpips_weight", 0.0) > 0:
        lpips_loss_fn = PerceptualLoss(device=device).to(device)
    
    # Setup Directories
    run_dir = ensure_dir(Path("experiments/runs") / args.run_id)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    tb_dir = ensure_dir(run_dir / "tb")
    recon_dir = ensure_dir(run_dir / "reconstructions")
    metrics_path = run_dir / "metrics.json"
    
    writer = SummaryWriter(log_dir=str(tb_dir))
    
    history = {"train_loss": [], "val_loss": [], "val_psnr": []}
    global_step = 0
    best_val = float("inf")
    epochs = run_cfg["epochs"]

    print(f"Starting training for {args.run_id} on {device}")

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        num_samples = 0
        
        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            
            # Forward
            recon, mu, logvar = model(images)
            
            # Loss
            loss, l1, kld, perc = vae_loss_fn(
                recon, images, mu, logvar, 
                kld_weight=run_cfg["kld_weight"],
                lpips_loss_fn=lpips_loss_fn,
                lpips_weight=run_cfg.get("lpips_weight", 0.0)
            )
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Logging
            bs = images.size(0)
            epoch_loss += loss.item() * bs
            epoch_recon += l1.item() * bs
            epoch_kld += kld.item() * bs
            num_samples += bs
            
            writer.add_scalar("loss/train_total", loss.item(), global_step)
            writer.add_scalar("loss/train_l1", l1.item(), global_step)
            writer.add_scalar("loss/train_kld", kld.item(), global_step)
            if run_cfg.get("lpips_weight", 0.0) > 0:
                writer.add_scalar("loss/train_lpips", perc.item(), global_step)
            global_step += 1
            
            if batch_idx % args.log_interval == 0:
                print(f"[{args.run_id}] Epoch {epoch:03d} Batch {batch_idx:05d} Loss: {loss.item():.4f} (L1: {l1.item():.4f}, KLD: {kld.item():.4f})")

        avg_train_loss = epoch_loss / max(1, num_samples)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_samples = 0
        sample_recon_saved = False
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                recon, mu, logvar = model(images)
                
                loss, _, _, _ = vae_loss_fn(
                    recon, images, mu, logvar, 
                    kld_weight=run_cfg["kld_weight"],
                    lpips_loss_fn=lpips_loss_fn,
                    lpips_weight=run_cfg.get("lpips_weight", 0.0)
                )
                
                bs = images.size(0)
                val_loss += loss.item() * bs
                val_psnr_sum += psnr(recon, images).sum().item()
                val_samples += bs
                
                # Save reconstruction grid
                if (not sample_recon_saved) and (epoch % args.vis_interval == 0 or epoch == epochs):
                    inp_vis = (images.clamp(-1, 1) + 1.0) * 0.5
                    rec_vis = (recon.clamp(-1, 1) + 1.0) * 0.5
                    grid = make_grid(torch.cat([inp_vis[:8], rec_vis[:8]], dim=0), nrow=8)
                    save_image(grid, recon_dir / f"epoch_{epoch:03d}.png")
                    writer.add_image("reconstructions/val", grid, epoch)
                    sample_recon_saved = True

        avg_val_loss = val_loss / max(1, val_samples)
        avg_val_psnr = val_psnr_sum / max(1, val_samples)
        
        # Update History
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)
        metrics_path.write_text(json.dumps(history, indent=2))
        
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
        writer.add_scalar("epoch/val_psnr", avg_val_psnr, epoch)
        
        print(f"[{args.run_id}] Epoch {epoch:03d}/{epochs:03d} Train={avg_train_loss:.4f} Val={avg_val_loss:.4f} PSNR={avg_val_psnr:.2f}")

        # --- CHECKPOINTING ---
        # Save Last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_id": args.run_id,
        }, checkpoints_dir / "last.pt")
        
        # Save Best
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "run_id": args.run_id,
            }, checkpoints_dir / "best.pt")

if __name__ == "__main__":
    main()