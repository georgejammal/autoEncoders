#!/bin/bash
set -e

for run_id in run_B2_resnet_lpips_w000 run_B2_resnet_lpips_w005 run_B2_resnet_lpips_w0075 run_B2_resnet_lpips_w010; do
  echo "Generating for $run_id..."
  /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/generate_reconstructions.py --run-id "$run_id" --mode best --num-images 10
done
