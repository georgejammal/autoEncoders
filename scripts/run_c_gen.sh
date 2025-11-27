#!/bin/bash
set -e

for run_id in run_C1_vae_beta_small run_C2_vae_beta_medium run_C3_vae_beta_large run_C4_vae_wide run_C5_vae_lpips; do
  echo "Generating for $run_id..."
  /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/generate_reconstructions.py --run-id "$run_id" --mode best --num-images 10
done

echo "All Part C reconstructions generated!"
