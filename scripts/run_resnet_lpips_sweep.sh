#!/usr/bin/env bash

# Launch the Deep ResNet Part B training four times:
#   1) No LPIPS
#   2) LPIPS with 5%
#   3) LPIPS with 7.5%
#   4) LPIPS with 10%
#
# Usage: ./scripts/run_resnet_lpips_sweep.sh [base_run_id]
# Default base_run_id is "run_B1_resnet_lpips".

set -euo pipefail

BASE_RUN_ID="${1:-run_B1_resnet_lpips}"
# Optional second argument: comma-separated GPU ids (default 0,1,2,3)
if [[ $# -ge 2 ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$2"
else
  GPU_IDS=(0 1 2 3)
fi

MODEL_MODULE="src.models.deep_resnet_autoencoder"
MODEL_CLASS="DeepResNetAutoencoder"
MODEL_KWARGS='{"latent_dim": 256}'

declare -a WEIGHTS=("0" "0.05" "0.075" "0.1")

if [[ ${#GPU_IDS[@]} -lt ${#WEIGHTS[@]} ]]; then
  echo "Need at least ${#WEIGHTS[@]} GPU ids; got ${#GPU_IDS[@]}" >&2
  exit 1
fi

pids=()
for idx in "${!WEIGHTS[@]}"; do
  weight="${WEIGHTS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  suffix="${weight//./}"
  if [[ "$weight" == "0" ]]; then
    run_id="${BASE_RUN_ID}_w000"
    echo ">>> [GPU ${gpu}] Training ${run_id} without LPIPS"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_part_b.py \
      --run-id "${run_id}" \
      --model-module "${MODEL_MODULE}" \
      --model-class "${MODEL_CLASS}" \
      --model-kwargs "${MODEL_KWARGS}" &
  else
    run_id="${BASE_RUN_ID}_w${suffix}"
    echo ">>> [GPU ${gpu}] Training ${run_id} with LPIPS weight=${weight}"
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_part_b.py \
      --run-id "${run_id}" \
      --model-module "${MODEL_MODULE}" \
      --model-class "${MODEL_CLASS}" \
      --model-kwargs "${MODEL_KWARGS}" \
      --use-lpips \
      --lpips-weight "${weight}" &
  fi
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
