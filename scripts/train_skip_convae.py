import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import yaml

from src.data.ffhq import build_dataloaders
from src.models import SkipConvAutoencoder, PerceptualLoss
from src.utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SkipConv autoencoder with L1+LPIPS")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse)


def train_one_epoch(model, loader, optimizer, device, l1_criterion, lpips_loss, lpips_weight):
    model.train()
    total_l1 = 0.0
    total_lpips = 0.0
    num_samples = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon = model(images)
        loss_l1 = l1_criterion(recon, images)
        loss_lpips = lpips_loss(recon, images)
        loss = loss_l1 + lpips_weight * loss_lpips

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_l1 += loss_l1.item() * bs
        total_lpips += loss_lpips.item() * bs
        num_samples += bs

    return total_l1 / num_samples, total_lpips / num_samples


def evaluate(model, loader, device, l1_criterion, lpips_loss):
    model.eval()
    total_l1 = 0.0
    total_lpips = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            recon = model(images)
            loss_l1 = l1_criterion(recon, images)
            loss_lpips = lpips_loss(recon, images)
            total_l1 += loss_l1.item() * images.size(0)
            total_lpips += loss_lpips.item() * images.size(0)
            total_psnr += psnr(recon, images).sum().item()
            num_samples += images.size(0)

    avg_l1 = total_l1 / num_samples
    avg_lpips = total_lpips / num_samples
    avg_psnr = total_psnr / num_samples
    return avg_l1, avg_lpips, avg_psnr


def plot_curves(metrics, out_path: Path):
    epochs = range(1, len(metrics["train_l1"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["train_l1"], label="Train L1")
    plt.plot(epochs, metrics["val_l1"], label="Val L1")
    plt.plot(epochs, metrics["val_lpips"], label="Val LPIPS")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training curves")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        root=Path(config["dataset_root"]),
        test_split=config.get("val_split", 1000),
        batch_size=config.get("batch_size", 16),
        num_workers=config.get("num_workers", 4),
        image_size=config.get("image_size", 256),
    )

    model = SkipConvAutoencoder(
        base_channels=config.get("base_channels", 32),
        num_blocks=config.get("num_blocks", 4),
        latent_dim=config.get("latent_dim", 256),
        output_activation=config.get("output_activation", "tanh"),
    ).to(device)

    optimizer = Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    l1_criterion = nn.L1Loss()
    lpips_loss = PerceptualLoss(device=device)
    lpips_weight = float(config.get("lpips_weight", 0.3))

    run_dir = ensure_dir(Path(config["run_dir"]))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    curves_path = Path(config.get("curves_path", "reports/curves")) / f"{run_dir.name}_loss.png"
    metrics_path = run_dir / "metrics.json"

    best_metric = float("inf")
    metrics = {"train_l1": [], "train_lpips": [], "val_l1": [], "val_lpips": [], "val_psnr": []}

    for epoch in range(1, config.get("epochs", 50) + 1):
        train_l1, train_lp = train_one_epoch(model, train_loader, optimizer, device, l1_criterion, lpips_loss, lpips_weight)
        val_l1, val_lp, val_psnr = evaluate(model, val_loader, device, l1_criterion, lpips_loss)

        metrics["train_l1"].append(train_l1)
        metrics["train_lpips"].append(train_lp)
        metrics["val_l1"].append(val_l1)
        metrics["val_lpips"].append(val_lp)
        metrics["val_psnr"].append(val_psnr)

        print(f"Epoch {epoch:03d}: train_l1={train_l1:.4f} val_l1={val_l1:.4f} val_lpips={val_lp:.4f} val_psnr={val_psnr:.2f}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoints_dir / "last.pt")

        total_val_metric = val_l1 + lpips_weight * val_lp
        if total_val_metric < best_metric:
            best_metric = total_val_metric
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoints_dir / "best.pt")

        metrics_path.write_text(json.dumps(metrics, indent=2))

    plot_curves(metrics, curves_path)


if __name__ == "__main__":
    main()
