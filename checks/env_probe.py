#!/usr/bin/env python3
import os
from pathlib import Path
import sys, os, site
try:
    usp = site.getusersitepackages()
    if isinstance(usp, str):
        sys.path = [p for p in sys.path if p != usp]
    else:
        for p in list(usp):
            if p in sys.path:
                sys.path.remove(p)
except Exception:
    pass
os.environ.setdefault("PYTHONNOUSERSITE", "1")
import torch
from PIL import Image
import numpy as np

DATASET_ROOT = Path("/home/ML_courses/03683533_2025/dataset")

def check_dataset():
    print("=== Dataset ===")
    if not DATASET_ROOT.exists():
        raise SystemExit(f"Missing dataset folder: {DATASET_ROOT}")

    samples = sorted(p.name for p in DATASET_ROOT.iterdir() if p.suffix.lower() == ".png")
    print(f"Total PNG files: {len(samples)}")
    if not samples:
        raise SystemExit("No PNG files found.")

    # Reserve first 1k for validation per spec
    val_split = samples[:1000]
    train_split = samples[1000:]
    print(f"Validation count (first 1000): {len(val_split)}")
    print(f"Training count (remaining): {len(train_split)}")

    sample_path = DATASET_ROOT / samples[0]
    img = Image.open(sample_path)
    arr = np.array(img)
    print(f"Sample {sample_path.name}: mode={img.mode}, size={img.size}, dtype={arr.dtype}")

def check_cuda():
    print("\n=== CUDA / PyTorch ===")
    print(f"PyTorch version: {torch.__version__}")
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        device_count = torch.cuda.device_count()
        print(f"GPU count: {device_count}")
        for idx in range(device_count):
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        print("No CUDA device detected.")

if __name__ == "__main__":
    check_dataset()
    check_cuda()