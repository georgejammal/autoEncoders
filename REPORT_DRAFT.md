# Homework Report: Autoencoder Implementation

## 1. Introduction
This report presents the implementation and analysis of Autoencoder models trained on the FFHQ dataset. The goal was to achieve high-quality image reconstruction and an informative latent space.

## 2. Methodology
We implemented several variations of Autoencoders and Variational Autoencoders (VAEs) using PyTorch.
- **Base Architecture**: ResNet-based Encoder and Decoder.
- **Variations**:
    - Skip Connections (U-Net style).
    - Variational Autoencoder (VAE) vs. Deterministic AE.
    - Loss Functions: L1, MSE, LPIPS, KL Divergence.
    - Attention Mechanisms.

## 3. Results Summary
The following table summarizes the performance of the trained models.

| Model Name | Type | Loss Function | Final Val Loss | LPIPS |
| :--- | :--- | :--- | :--- | :--- |
| *[Table to be populated]* | | | | |

## 4. Best Model Analysis
**Model ID:** `[Best Model ID]`
**Architecture:** `[Architecture Details]`
**Hyperparameters:** `[Hyperparams]`

### Qualitative Results
![Best Model Reconstructions](experiments/runs/[Best Model ID]/reconstructions/epoch_40.png)

### Quantitative Results
- **Validation Loss:** `[Val Loss]`
- **LPIPS Score:** `[LPIPS]`

## 5. Comparative Analysis (10 Other Models)
We compared the best model against 10 other variations to understand the impact of different components.

### 5.1 Ablation Study
- **Effect of VAE vs AE:** ...
- **Effect of LPIPS Loss:** ...
- **Effect of Attention:** ...

### 5.2 Loss Curves
![Training Loss Curves](loss_curves.png)

## 6. Conclusion
[Summary of findings]
