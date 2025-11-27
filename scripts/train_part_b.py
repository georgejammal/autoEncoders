import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the project root (containing `src/`) is on sys.path so that
# `import src.*` works even when this script is launched from `scripts/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force line-buffered stdout/stderr so training logs flush immediately
# even when redirected to files or screens.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Store Torch Hub downloads inside the repository to avoid hitting the
# limited per-user home quota.
TORCH_HUB_DIR = PROJECT_ROOT / ".cache" / "torch" / "hub"
TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_HUB_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
torch.hub.set_dir(str(TORCH_HUB_DIR))


from src.data.ffhq import build_dataloaders
from src.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Part B autoencoders with a user-defined architecture."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID used only for naming the output directory under experiments/runs/.",
    )
    parser.add_argument(
        "--model-module",
        type=str,
        default="src.models.deep_resnet_autoencoder",
        help="Python module where your model class is defined (default: 'src.models.deep_resnet_autoencoder').",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="DeepResNetAutoencoder",
        help="Model class name inside the module (default: 'DeepResNetAutoencoder').",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON dict with keyword arguments passed to the model constructor.",
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
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate.",
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
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="If >0, limit the number of train batches per epoch (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="If >0, limit the number of validation batches per epoch.",
    )
    parser.add_argument(
        "--use-lpips",
        action="store_true",
        help="Add an LPIPS perceptual term to the training and validation loss.",
    )
    parser.add_argument(
        "--lpips-weight",
        type=float,
        default=1.0,
        help="Weight for the LPIPS loss term (only used when --use-lpips is set).",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone network for LPIPS when --use-lpips is enabled.",
    )
    return parser.parse_args()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def build_model(module_path: str, class_name: str, model_kwargs: dict) -> nn.Module:
    """
    Dynamically import your model and construct it.

    Defaults point to src/models/deep_resnet_autoencoder.py::DeepResNetAutoencoder,
    but you can swap module_path and class_name via command-line arguments.
    """
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**model_kwargs)


def main() -> None:
    args = parse_args()
    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse --model-kwargs JSON: {exc}")

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None

    lpips_loss_fn = None
    lpips_weight = float(args.lpips_weight)
    if args.use_lpips:
        try:
            import lpips  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - import-time error path
            raise SystemExit(
                "You passed --use-lpips but the 'lpips' package is not installed.\n"
                "Install it with 'pip install lpips' inside your course conda environment."
            ) from exc
        lpips_loss_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_loss_fn.eval()
        for p in lpips_loss_fn.parameters():
            p.requires_grad_(False)

    train_loader, val_loader = build_dataloaders(
        root=args.dataset_root,
        test_split=args.val_split,
        batch_size=batch_size,
        num_workers=args.num_workers,
        image_size=256,
    )

    model = build_model(args.model_module, args.model_class, model_kwargs).to(device)
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
        epoch_train_lpips = 0.0
        num_train_samples = 0
        epoch_start_time = time.time()

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            recon = model(images)
            loss = criterion(recon, images)
            lpips_term = torch.tensor(0.0, device=device)
            if lpips_loss_fn is not None:
                lpips_term = lpips_loss_fn(images, recon).mean()
                loss = loss + lpips_weight * lpips_term

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            epoch_train_loss += loss.item() * bs
            epoch_train_lpips += lpips_term.item() * bs
            num_train_samples += bs

            writer.add_scalar("loss/train_l1", loss.item(), global_step)
            if lpips_loss_fn is not None:
                writer.add_scalar("loss/train_lpips", lpips_term.item(), global_step)
            global_step += 1

            if batch_idx % args.log_interval == 0:
                total_batches = len(train_loader)
                if max_train_batches is not None:
                    total_batches = min(total_batches, max_train_batches)
                elapsed = time.time() - epoch_start_time
                print(
                    f"[{args.run_id}] Epoch {epoch:03d} "
                    f"Batch {batch_idx:05d}/{total_batches:05d} "
                    f"Train L1: {loss.item():.4f} "
                    f"Elapsed {elapsed:7.2f}s"
                )
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        avg_train_loss = epoch_train_loss / max(1, num_train_samples)

        model.eval()
        val_loss = 0.0
        val_lpips = 0.0
        val_psnr_sum = 0.0
        val_samples = 0

        sample_recon_saved = False
        with torch.no_grad():
            for val_batch_idx, (images, _) in enumerate(val_loader, start=1):
                images = images.to(device, non_blocking=True)
                recon = model(images)
                loss = criterion(recon, images)
                lpips_val = torch.tensor(0.0, device=device)
                if lpips_loss_fn is not None:
                    lpips_val = lpips_loss_fn(images, recon).mean()
                    loss = loss + lpips_weight * lpips_val

                bs = images.size(0)
                val_loss += loss.item() * bs
                val_lpips += lpips_val.item() * bs
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
                if max_val_batches is not None and val_batch_idx >= max_val_batches:
                    break

        avg_val_loss = val_loss / max(1, val_samples)
        avg_val_psnr = val_psnr_sum / max(1, val_samples)
        avg_train_lpips = epoch_train_lpips / max(1, num_train_samples)
        avg_val_lpips = val_lpips / max(1, val_samples)

        history["train_l1"].append(avg_train_loss)
        history["val_l1"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)
        if lpips_loss_fn is not None:
            history.setdefault("train_lpips", []).append(avg_train_lpips)
            history.setdefault("val_lpips", []).append(avg_val_lpips)
        metrics_path.write_text(json.dumps(history, indent=2))

        writer.add_scalar("epoch/train_l1", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_l1", avg_val_loss, epoch)
        writer.add_scalar("epoch/val_psnr", avg_val_psnr, epoch)
        if lpips_loss_fn is not None:
            writer.add_scalar("epoch/train_lpips", avg_train_lpips, epoch)
            writer.add_scalar("epoch/val_lpips", avg_val_lpips, epoch)

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
