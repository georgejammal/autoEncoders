from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Residual Block (BN + LeakyReLU)
# ---------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.block(x) + x, negative_slope=0.2, inplace=True)


# ---------------------------------------------------------
# ENCODER
# ---------------------------------------------------------
class ResNetEncoder(nn.Module):
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

        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


# ---------------------------------------------------------
# VAE ENCODER (mu, logvar)
# ---------------------------------------------------------
class ResNetEncoderVAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )
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
        self.down4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(512),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# ---------------------------------------------------------
# DECODER
# ---------------------------------------------------------
class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.unflatten = nn.Unflatten(1, (512, 8, 8))

        self.up0 = nn.Sequential(
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up1 = nn.Sequential(
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up2 = nn.Sequential(
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up3 = nn.Sequential(
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up4 = nn.Sequential(
            ResBlock(128),
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
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = torch.tanh(self.final(x))
        return x


# ---------------------------------------------------------
# FULL AUTOENCODER (plain)
# ---------------------------------------------------------
class DeepResNetAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ---------------------------------------------------------
# VAE VARIANT (KL-regularized)
# ---------------------------------------------------------
class DeepResNetVAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoderVAE(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)

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
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------
# VECTOR QUANTIZER (VQ-VAE)
# ---------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D)
        flat_z = z.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices)  # (B*1, D)
        z_q = z_q.view_as(z)

        # VQ-VAE losses
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        loss = e_latent_loss + self.commitment_cost * q_latent_loss

        # Straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, loss, encoding_indices.view(z.shape[0], -1)


class DeepResNetVQAutoencoder(nn.Module):
    """
    VQ-VAE style autoencoder where the continuous latent
    vector (size 256) is quantized using a codebook.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = ResNetDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss


# ---------------------------------------------------------
# DIFFUSION-LIKE DENOISING AUTOENCODER
# ---------------------------------------------------------
class DiffusionLikeAutoencoder(nn.Module):
    """
    Simple diffusion-like autoencoder:
    - During training, you can add noise to the input
      and condition the latent on a noise level scalar.
    - Architecture reuses the ResNet encoder/decoder with
      a 256-D latent space.
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor, noise_level: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(x)
        if noise_level is not None:
            if noise_level.dim() == 1:
                noise_level = noise_level.view(-1, 1)
            t_embed = self.time_mlp(noise_level.to(z.dtype))
            z = z + t_embed
        recon = self.decoder(z)
        return recon
