from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Self-Attention Block
# ---------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Self-attention block for spatial features.
    Uses multi-head attention with residual connection.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Ensure channels is divisible by num_heads
        assert channels % num_heads == 0, f"channels={channels} must be divisible by num_heads={num_heads}"
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Normalize and compute Q, K, V
        h = self.norm(x)
        qkv = self.qkv(h)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # (B, num_heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project and residual
        out = self.proj(out)
        return x + out


# ---------------------------------------------------------
# Residual Block (Improved with GroupNorm)
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Using GroupNorm(32) instead of BatchNorm as requested
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.block(x) + x, negative_slope=0.2, inplace=True)


# ---------------------------------------------------------
# ENCODER with Attention (for VAE)
# ---------------------------------------------------------
class ResNetEncoderVAEAttention(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64), # Keep BN in stem if needed, or switch to GN. Sticking to BN for stem as per notes "Keep BN only in the initial downsampling convs if needed"
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            ResBlock(128),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            ResBlock(256),
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
            ResBlock(512),
        )
        
        # Attention at 32x32 resolution
        self.attn_32 = SelfAttention(512, num_heads=8)
        
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        
        # Attention at 16x16 resolution
        self.attn_16 = SelfAttention(512, num_heads=8)
        
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        
        # Global Average Pooling instead of Flatten
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # FC from 512 (channels) to latent_dim
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.attn_32(x)  # Attention at 32x32
        x = self.down4(x)
        x = self.attn_16(x)  # Attention at 16x16
        x = self.down5(x)
        
        # GAP + Flatten
        x = self.global_pool(x)
        x = self.flatten(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# ---------------------------------------------------------
# ENCODER with Attention (for AE)
# ---------------------------------------------------------
class ResNetEncoderAttention(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(128),
            ResBlock(128),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(256),
            ResBlock(256),
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
            ResBlock(512),
        )
        
        # Attention at 32x32 resolution
        self.attn_32 = SelfAttention(512, num_heads=8)
        
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        
        # Attention at 16x16 resolution
        self.attn_16 = SelfAttention(512, num_heads=8)
        
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.attn_32(x)  # Attention at 32x32
        x = self.down4(x)
        x = self.attn_16(x)  # Attention at 16x16
        x = self.down5(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


# ---------------------------------------------------------
# DECODER with Attention (Strengthened)
# ---------------------------------------------------------
class ResNetDecoderAttention(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.unflatten = nn.Unflatten(1, (512, 8, 8))
        
        self.up0 = nn.Sequential(
            ResBlock(512),
            ResBlock(512), # Extra ResBlock
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Attention at 16x16 resolution
        self.attn_16 = SelfAttention(512, num_heads=8)
        
        self.up1 = nn.Sequential(
            ResBlock(512),
            ResBlock(512), # Extra ResBlock
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Attention at 32x32 resolution
        self.attn_32 = SelfAttention(512, num_heads=8)
        
        self.up2 = nn.Sequential(
            ResBlock(512),
            ResBlock(512), # Extra ResBlock
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256), # Extra ResBlock
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up4 = nn.Sequential(
            ResBlock(128),
            ResBlock(128), # Extra ResBlock
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.up0(x)
        x = self.attn_16(x)  # Attention at 16x16
        x = self.up1(x)
        x = self.attn_32(x)  # Attention at 32x32
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = torch.tanh(self.final(x))
        return x


# ---------------------------------------------------------
# FULL MODELS
# ---------------------------------------------------------
class DeepResNetAttentionVAE(nn.Module):
    """Deep ResNet VAE with Self-Attention blocks."""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoderVAEAttention(latent_dim)
        self.decoder = ResNetDecoderAttention(latent_dim)
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class DeepResNetAttentionAE(nn.Module):
    """Deep ResNet Autoencoder with Self-Attention blocks."""
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoderAttention(latent_dim)
        self.decoder = ResNetDecoderAttention(latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
