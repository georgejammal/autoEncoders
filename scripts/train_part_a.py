import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# Ensure the project root (containing `src/`) is on sys.path so that
# `import src.*` works even when this script is launched from `scripts/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ffhq import build_dataloaders
from src.models import SkipConvAutoencoder
from src.utils import ensure_dir, set_seed


PART_A_RUNS = {
    "run_A1_skip_convae_depth10": {
        "model_kwargs": {
            "base_channels": 32,
            "num_blocks": 4,
            "latent_dim": 256,
            "output_activation": "tanh",
            "norm": "group",
            "activation": "silu",
        },
        "batch_size": 8,
        "epochs": 60,
        "learning_rate": 2e-4,
    },
    "run_A2_skip_convae_bn": {
        "model_kwargs": {
            "base_channels": 32,
            "num_blocks": 4,
            "latent_dim": 256,
            "output_activation": "tanh",
            "norm": "batch",
            "activation": "silu",
        },
        "batch_size": 8,
        "epochs": 60,
        "learning_rate": 2e-4,
    },
    "run_A3_skip_convae_leaky": {
        "model_kwargs": {
            "base_channels": 32,
            "num_blocks": 4,
            "latent_dim": 256,
            "output_activation": "tanh",
            "norm": "group",
            "activation": "leaky_relu",
        },
        "batch_size": 8,
        "epochs": 60,
        "learning_rate": 2e-4,
    },
    "run_A4_skip_convae_wide": {
        "model_kwargs": {
            "base_channels": 48,
            "num_blocks": 4,
            "latent_dim": 256,
            "output_activation": "tanh",
            "norm": "group",
            "activation": "silu",
        },
        "batch_size": 8,
        "epochs": 60,
        "learning_rate": 2e-4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Part A skip-conv autoencoders.")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        choices=sorted(PART_A_RUNS.keys()),
        help="Run ID corresponding to Part A experiment.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ML_courses/03683533_2025/dataset"),
        help="Path to FFHQ dataset root.",
    )
    parser.add_argument(
        "--val-split",
        type=int,
        default=1000,
        help="Number of validation images (first N).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (defaults from run config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (defaults from run config).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate (defaults from run config).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Batches between log prints.",
    )
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=15,
        help="Epoch interval for saving reconstructions.",
    )
    return parser.parse_args()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def main() -> None:
    args = parse_args()
    run_cfg = PART_A_RUNS[args.run_id]

    batch_size = args.batch_size or run_cfg["batch_size"]
    epochs = args.epochs or run_cfg["epochs"]
    learning_rate = args.learning_rate or run_cfg["learning_rate"]

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        root=args.dataset_root,
        test_split=args.val_split,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=256,
    )

    model = SkipConvAutoencoder(**run_cfg["model_kwargs"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    run_dir = ensure_dir(Path("experiments/runs") / args.run_id)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    tb_dir = ensure_dir(run_dir / "tb")
    recon_dir = ensure_dir(run_dir / "reconstructions")
    metrics_path = run_dir / "metrics.json"

    writer = SummaryWriter(log_dir=str(tb_dir))

    history = {
        "train_l1": [],
        "val_l1": [],
        "val_psnr": [],
    }

    global_step = 0
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_train_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            recon = model(images)
            loss = criterion(recon, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            epoch_train_loss += loss.item() * bs
            num_train_samples += bs

            writer.add_scalar("loss/train_l1", loss.item(), global_step)
            global_step += 1

            if batch_idx % args.log_interval == 0:
                print(
                    f"[{args.run_id}] Epoch {epoch:03d} "
                    f"Batch {batch_idx:05d}/{len(train_loader):05d} "
                    f"Train L1: {loss.item():.4f}"
                )

        avg_train_loss = epoch_train_loss / max(1, num_train_samples)

        model.eval()
        val_loss = 0.0
        val_psnr_sum = 0.0
        val_samples = 0

        sample_recon_saved = False
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                recon = model(images)
                loss = criterion(recon, images)

                bs = images.size(0)
                val_loss += loss.item() * bs
                val_psnr_sum += psnr(recon, images).sum().item()
                val_samples += bs

                # Save a reconstruction grid for the first validation batch
                if (not sample_recon_saved) and (epoch % args.vis_interval == 0 or epoch == epochs):
                    # Denormalize from [-1, 1] to [0, 1] for visualization
                    inp_vis = (images.clamp(-1, 1) + 1.0) * 0.5
                    rec_vis = (recon.clamp(-1, 1) + 1.0) * 0.5
                    grid = make_grid(
                        torch.cat([inp_vis[:8], rec_vis[:8]], dim=0),
                        nrow=8,
                    )
                    save_path = recon_dir / f"epoch_{epoch:03d}.png"
                    save_image(grid, save_path)
                    writer.add_image("reconstructions/input_vs_recon", grid, epoch)
                    sample_recon_saved = True

        avg_val_loss = val_loss / max(1, val_samples)
        avg_val_psnr = val_psnr_sum / max(1, val_samples)

        history["train_l1"].append(avg_train_loss)
        history["val_l1"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)
        metrics_path.write_text(json.dumps(history, indent=2))

        writer.add_scalar("epoch/train_l1", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_l1", avg_val_loss, epoch)
        writer.add_scalar("epoch/val_psnr", avg_val_psnr, epoch)

        print(
            f"[{args.run_id}] Epoch {epoch:03d}/{epochs:03d} "
            f"Train L1={avg_train_loss:.4f} "
            f"Val L1={avg_val_loss:.4f} "
            f"Val PSNR={avg_val_psnr:.2f}"
        )

        # Always save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "run_id": args.run_id,
            },
            checkpoints_dir / "last.pt",
        )

        # Periodic checkpoints every 15 epochs (and at the end)
        if epoch % 15 == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_id": args.run_id,
                },
                checkpoints_dir / f"epoch_{epoch:03d}.pt",
            )

        # Track best based on validation L1
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "run_id": args.run_id,
                },
                checkpoints_dir / "best.pt",
            )

    writer.close()


if __name__ == "__main__":
    main()
