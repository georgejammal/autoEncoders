#!/usr/bin/env bash

# Launch the four Part A runs in parallel, one per GPU.

set -e

python --version

CUDA_VISIBLE_DEVICES=0 python scripts/train_part_a.py --run-id run_A1_skip_convae_depth10 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_part_a.py --run-id run_A2_skip_convae_bn &
CUDA_VISIBLE_DEVICES=2 python scripts/train_part_a.py --run-id run_A3_skip_convae_leaky &
CUDA_VISIBLE_DEVICES=3 python scripts/train_part_a.py --run-id run_A4_skip_convae_wide &

wait

