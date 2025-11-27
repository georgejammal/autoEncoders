# Part D - Deep VAE Models with Attention

## Overview

This implementation includes 4 new autoencoder models designed to achieve an informative latent space while maintaining high-quality reconstruction:

### Models

1. **run_D1_deep_resnet_vae** - Deep ResNet VAE (Baseline)
   - Architecture: `DeepResNetVAE` from `deep_resnet_autoencoder.py`
   - Loss: L1 + KL Divergence (weight=0.00025)
   - Purpose: Baseline deep VAE

2. **run_D2_deep_resnet_vae_lpips** - Deep ResNet VAE + LPIPS
   - Architecture: `DeepResNetVAE`
   - Loss: L1 + KL Divergence + LPIPS (weight=0.1)
   - Purpose: Test if LPIPS improves VAE reconstruction quality

3. **run_D3_attention_ae** - Deep ResNet AE + Attention
   - Architecture: `DeepResNetAttentionAE` (NEW)
   - Loss: L1 only (no KL divergence - this is an AE, not VAE)
   - Purpose: Test if attention improves reconstruction without VAE regularization

4. **run_D4_attention_vae_lpips** - Deep ResNet VAE + Attention + LPIPS
   - Architecture: `DeepResNetAttentionVAE` (NEW)
   - Loss: L1 + KL Divergence + LPIPS
   - Purpose: Best of all worlds - attention + VAE + perceptual loss

### Attention Mechanism

The attention models (`DeepResNetAttentionAE` and `DeepResNetAttentionVAE`) include:
- Self-attention blocks at 16x16 and 32x32 resolutions
- Multi-head attention (8 heads) for capturing global dependencies
- Residual connections to preserve local information
- No U-Net skip connections (as requested)

## Training Configuration

### Memory & Time Constraints
- **Batch Size**: 4 (reduced from 8 to fit 12GB GPU memory with attention)
- **Epochs**: 40 (estimated ~10 hours total on 4 GPUs in parallel)
- **Learning Rate**: 2e-4 (Adam optimizer)
- **Image Size**: 256x256
- **Latent Dim**: 256 (flattened)

### Loss Tracking
All metrics are saved in the same format as Part A and Part B:
- `experiments/runs/{run_id}/metrics.json` - JSON file with loss history
- `experiments/runs/{run_id}/tb/` - TensorBoard logs
- `experiments/runs/{run_id}/reconstructions/` - Visual reconstructions every 10 epochs
- `experiments/runs/{run_id}/checkpoints/` - Model checkpoints (best, last, periodic)

## How to Run

### Launch All 4 Models in Parallel
```bash
cd /home/ML_courses/03683533_2025/ameer_george_abdallah
./scripts/launch_part_d.sh
```

This will start 4 training jobs, one on each GPU (0-3).

### Monitor Progress
```bash
# Watch logs in real-time
tail -f logs/run_D1_deep_resnet_vae.log
tail -f logs/run_D2_deep_resnet_vae_lpips.log
tail -f logs/run_D3_attention_ae.log
tail -f logs/run_D4_attention_vae_lpips.log

# Check GPU usage
nvidia-smi

# View TensorBoard
tensorboard --logdir experiments/runs --port 6006
```

### Run Individual Models
```bash
# Example: Run only Model 1
CUDA_VISIBLE_DEVICES=0 python scripts/train_part_d.py \
    --run-id "run_D1_deep_resnet_vae" \
    --model-module "src.models.deep_resnet_autoencoder" \
    --model-class "DeepResNetVAE" \
    --model-kwargs '{"latent_dim": 256}' \
    --is-vae \
    --kld-weight 0.00025 \
    --batch-size 4 \
    --epochs 40 \
    --learning-rate 2e-4
```

## Files Created/Modified

### New Files
- `src/models/vae_attention.py` - Attention-based VAE and AE models
- `scripts/train_part_d.py` - Training script supporting both VAE and AE
- `scripts/launch_part_d.sh` - Parallel launch script

### Modified Files
- `src/models/__init__.py` - Added new model exports

## Expected Results

After training completes (~10 hours), you will have:
- 4 trained models with different architectures and loss functions
- Loss curves comparing VAE vs AE, with/without attention, with/without LPIPS
- Visual reconstructions showing quality differences
- Checkpoints for the best models based on validation loss

This provides a comprehensive ablation study to determine the best approach for achieving both an informative latent space (VAE) and high-quality decoder (Attention + LPIPS).
