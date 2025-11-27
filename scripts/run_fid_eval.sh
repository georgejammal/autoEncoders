#!/bin/bash
# Script to calculate FID for Part D models

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# 1. Install torch-fidelity if needed
echo "Checking/Installing torch-fidelity..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python -m pip install torch-fidelity

# 2. Evaluate D1 (Deep ResNet VAE)
echo "Evaluating D1 (Generation Mode)..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/calc_fid.py \
    --run-id run_D1_deep_resnet_vae \
    --mode generation \
    --num-samples 1000

echo "Evaluating D1 (Reconstruction Mode)..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/calc_fid.py \
    --run-id run_D1_deep_resnet_vae \
    --mode reconstruction \
    --num-samples 1000

# 3. Evaluate D2 (Deep ResNet VAE + LPIPS)
echo "Evaluating D2 (Generation Mode)..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/calc_fid.py \
    --run-id run_D2_deep_resnet_vae_lpips \
    --mode generation \
    --num-samples 1000

echo "Evaluating D2 (Reconstruction Mode)..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/calc_fid.py \
    --run-id run_D2_deep_resnet_vae_lpips \
    --mode reconstruction \
    --num-samples 1000

# 4. Evaluate D3 (Attention AE)
echo "Evaluating D3 (Reconstruction Mode)..."
/home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/calc_fid.py \
    --run-id run_D3_attention_ae \
    --mode reconstruction \
    --num-samples 1000

echo "FID Evaluation Complete!"
