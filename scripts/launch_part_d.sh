#!/bin/bash
# Launch script for Part D experiments
# Runs 4 models in parallel on 4 GPUs

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Model 1: Deep ResNet VAE (Baseline)
CUDA_VISIBLE_DEVICES=0 nohup /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/train_part_d.py \
    --run-id "run_D1_deep_resnet_vae" \
    --model-module "src.models.deep_resnet_autoencoder" \
    --model-class "DeepResNetVAE" \
    --model-kwargs '{"latent_dim": 256}' \
    --is-vae \
    --kld-weight 0.00025 \
    --batch-size 4 \
    --epochs 30 \
    --learning-rate 2e-4 \
    > logs/run_D1_deep_resnet_vae.log 2>&1 &

# Model 2: Deep ResNet VAE + LPIPS
CUDA_VISIBLE_DEVICES=1 nohup /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/train_part_d.py \
    --run-id "run_D2_deep_resnet_vae_lpips" \
    --model-module "src.models.deep_resnet_autoencoder" \
    --model-class "DeepResNetVAE" \
    --model-kwargs '{"latent_dim": 256}' \
    --is-vae \
    --kld-weight 0.00025 \
    --use-lpips \
    --lpips-weight 0.1 \
    --batch-size 4 \
    --epochs 30 \
    --learning-rate 2e-4 \
    > logs/run_D2_deep_resnet_vae_lpips.log 2>&1 &

# Model 3: Deep ResNet AE + Attention (NO VAE)
CUDA_VISIBLE_DEVICES=2 nohup /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/train_part_d.py \
    --run-id "run_D3_attention_ae" \
    --model-module "src.models.vae_attention" \
    --model-class "DeepResNetAttentionAE" \
    --model-kwargs '{"latent_dim": 256}' \
    --batch-size 4 \
    --epochs 30 \
    --learning-rate 2e-4 \
    > logs/run_D3_attention_ae.log 2>&1 &

# Model 4: Deep ResNet VAE + Attention + LPIPS
CUDA_VISIBLE_DEVICES=3 nohup /home/ML_courses/03683533_2025/ameer_george_abdallah/envs/ae38/bin/python scripts/train_part_d.py \
    --run-id "run_D4_attention_vae_lpips" \
    --model-module "src.models.vae_attention" \
    --model-class "DeepResNetAttentionVAE" \
    --model-kwargs '{"latent_dim": 256}' \
    --is-vae \
    --kld-weight 0.00025 \
    --use-lpips \
    --lpips-weight 0.1 \
    --batch-size 4 \
    --epochs 30 \
    --learning-rate 2e-4 \
    > logs/run_D4_attention_vae_lpips.log 2>&1 &

echo "Launched 4 training jobs in parallel on GPUs 0-3"
echo "Monitor progress with:"
echo "  tail -f logs/run_D1_deep_resnet_vae.log"
echo "  tail -f logs/run_D2_deep_resnet_vae_lpips.log"
echo "  tail -f logs/run_D3_attention_ae.log"
echo "  tail -f logs/run_D4_attention_vae_lpips.log"
echo ""
echo "Check GPU usage with: nvidia-smi"
echo ""
echo "Wait for all jobs to complete:"
echo "  wait"
