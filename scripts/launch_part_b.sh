#!/usr/bin/env bash

# Launch the four Part B ResNet-family runs in parallel, one per GPU.

set -e

python --version

CUDA_VISIBLE_DEVICES=0 python scripts/train_part_b_models.py --run-id run_B1_resnet_ae &
CUDA_VISIBLE_DEVICES=1 python scripts/train_part_b_models.py --run-id run_B2_resnet_vae &
CUDA_VISIBLE_DEVICES=2 python scripts/train_part_b_models.py --run-id run_B3_resnet_vq &
CUDA_VISIBLE_DEVICES=3 python scripts/train_part_b_models.py --run-id run_B4_diffusion_like &

wait

