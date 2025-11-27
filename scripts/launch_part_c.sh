#!/usr/bin/env bash

# Launch the four Part C VAE experiments in parallel, one per GPU.

set -e

echo "Starting Part C VAE Experiments..."

CUDA_VISIBLE_DEVICES=0 python scripts/train_part_c.py --run-id run_C1_vae_beta_small &
CUDA_VISIBLE_DEVICES=1 python scripts/train_part_c.py --run-id run_C2_vae_beta_medium &
CUDA_VISIBLE_DEVICES=2 python scripts/train_part_c.py --run-id run_C3_vae_beta_large &
CUDA_VISIBLE_DEVICES=3 python scripts/train_part_c.py --run-id run_C4_vae_wide &

wait