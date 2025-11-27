from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def depthwise_separable_conv(
    in_ch: int,
    out_ch: int,
    stride: int = 1,
    norm: str = "group",
    activation: str = "silu",
) -> nn.Sequential:
    """Depthwise + pointwise conv block with configurable norm and activation."""
    layers = []

    # depthwise conv
    layers.append(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_ch,
            bias=False,
        )
    )
    if norm == "batch":
        layers.append(nn.BatchNorm2d(in_ch))
    else:
        num_groups_in = max(1, in_ch // 8)
        layers.append(nn.GroupNorm(num_groups=num_groups_in, num_channels=in_ch))

    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.SiLU(inplace=True))

    # pointwise conv
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
    if norm == "batch":
        layers.append(nn.BatchNorm2d(out_ch))
    else:
        num_groups_out = max(1, out_ch // 8)
        layers.append(nn.GroupNorm(num_groups=num_groups_out, num_channels=out_ch))

    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    else:
        layers.append(nn.SiLU(inplace=True))

    return nn.Sequential(*layers)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "group", activation: str = "silu"):
        super().__init__()
        self.block = depthwise_separable_conv(in_ch, out_ch, norm=norm, activation=activation)
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        feat = self.block(x)
        return feat, self.down(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "group", activation: str = "silu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.fuse = depthwise_separable_conv(out_ch + skip_ch, out_ch, norm=norm, activation=activation)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class SkipConvAutoencoder(nn.Module):
    """Conv autoencoder with skip connections and 256-D latent code."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_blocks: int = 4,
        latent_dim: int = 256,
        output_activation: str = "tanh",
        norm: str = "group",
        activation: str = "silu",
    ):
        super().__init__()
        assert num_blocks >= 3
        self.stem = depthwise_separable_conv(
            in_channels,
            base_channels,
            norm=norm,
            activation=activation,
        )

        enc_modules = []
        enc_channels = []
        in_ch = base_channels
        for i in range(num_blocks):
            out_ch = base_channels * (2 ** (i + 1))
            enc_modules.append(EncoderBlock(in_ch, out_ch, norm=norm, activation=activation))
            enc_channels.append(out_ch)
            in_ch = out_ch
        self.encoders = nn.ModuleList(enc_modules)

        bottleneck_channels = enc_channels[-1]
        self.bottleneck = depthwise_separable_conv(
            bottleneck_channels,
            bottleneck_channels,
            norm=norm,
            activation=activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(bottleneck_channels, latent_dim)
        self.from_latent = nn.Linear(latent_dim, bottleneck_channels)
        self.expand = depthwise_separable_conv(
            bottleneck_channels,
            bottleneck_channels,
            norm=norm,
            activation=activation,
        )

        dec_modules = []
        decoder_in = bottleneck_channels
        for skip_ch in reversed(enc_channels[:-1]):
            dec_modules.append(DecoderBlock(decoder_in, skip_ch, skip_ch, norm=norm, activation=activation))
            decoder_in = skip_ch
        self.decoders = nn.ModuleList(dec_modules)
        self.final_up = DecoderBlock(decoder_in, base_channels, base_channels, norm=norm, activation=activation)

        self.head = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        if output_activation == "tanh":
            self.out_act = nn.Tanh()
        elif output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)
        for enc in self.encoders:
            feat, x = enc(x)
            skips.append(feat)

        x = self.bottleneck(x)
        b = x.shape[0]
        latent = self.avgpool(x).view(b, -1)
        latent = self.to_latent(latent)
        x = self.from_latent(latent).view(b, -1, 1, 1)
        x = self.expand(x)

        skip_iter = list(reversed(skips[:-1]))
        for dec, skip in zip(self.decoders, skip_iter):
            x = dec(x, skip)

        x = self.final_up(x, skips[0])
        x = self.head(x)
        return self.out_act(x)
