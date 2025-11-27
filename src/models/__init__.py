"""Model registry for running multiple autoencoder variants."""

from .convae import ConvAutoencoder
from .skip_convae import SkipConvAutoencoder
from .unet import UNetAutoencoder, PerceptualLoss
from .vae import VAE
from .vae_attention import DeepResNetAttentionVAE, DeepResNetAttentionAE

__all__ = [
    "ConvAutoencoder",
    "SkipConvAutoencoder",
    "UNetAutoencoder",
    "PerceptualLoss",
    "VAE",
    "DeepResNetAttentionVAE",
    "DeepResNetAttentionAE",
]
