import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # up-convert channels then concat with skip; final conv reduces to out_ch
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (odd sizes)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """
    U-Net autoencoder with skip connections.
    - Fully convolutional, shape-agnostic (pads to multiples of 2^depth, then unpads).
    - Uses a conv bottleneck that reduces channels to latent_dim via 1x1 convs.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        latent_dim: int = 256,
        out_activation: str = "tanh",  # "tanh" to match [-1,1] normalization
    ):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"

        # Encoder channel plan
        chs = [base_channels * (2 ** i) for i in range(depth + 1)]  # depth+1 gives bottom stage
        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])

        # Bottleneck channel compression to latent_dim (conv-only)
        self.bottleneck_reduce = nn.Conv2d(chs[-1], latent_dim, kernel_size=1)
        self.bottleneck_expand = nn.Conv2d(latent_dim, chs[-1], kernel_size=1)

        # Decoder: up modules
        ups_in = list(reversed(chs[1:]))  # encoder outputs excluding inc, reversed
        self.ups = nn.ModuleList([Up(ups_in[i], ups_in[i + 1]) for i in range(len(ups_in) - 1)])
        # Final up to base_channels
        self.up_last = Up(ups_in[-1], chs[0])

        # Final projection
        self.outc = nn.Conv2d(chs[0], in_channels, kernel_size=1)
        if out_activation == "tanh":
            self.out_act = nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Identity()

        self.depth = depth

    def _pad_to_multiple(self, x, multiple=16):
        b, c, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        # pad order: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (0, pad_w, 0, pad_h)

    def _unpad(self, x, pad):
        _, pad_w, _, pad_h = pad
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., : x.size(-2) - pad_h, : x.size(-1) - pad_w]

    def forward(self, x):
        # ensure divisibility by 2^depth
        scale = 2 ** self.depth
        x_pad, pad = self._pad_to_multiple(x, multiple=scale)

        # encoder
        x1 = self.inc(x_pad)
        skips = [x1]
        x_enc = x1
        for down in self.downs:
            x_enc = down(x_enc)
            skips.append(x_enc)

        # bottleneck conv latent compression
        x_enc = self.bottleneck_reduce(x_enc)
        x_enc = F.relu(x_enc, inplace=True)
        x_enc = self.bottleneck_expand(x_enc)
        x_enc = F.relu(x_enc, inplace=True)

        # decoder with skips (skip list: deepest to shallowest excluding last bottleneck)
        # skips currently: [enc0, enc1, ..., enc_depth]
        # start from bottleneck output; pair with skips[-2], then skips[-3], ...
        x_dec = x_enc
        skip_iter = list(reversed(skips[:-1]))  # drop deepest because it's matched by current stage
        for i, up in enumerate(self.ups):
            x_dec = up(x_dec, skip_iter[i])

        # last up to base_channels using the very first encoder output
        x_dec = self.up_last(x_dec, skips[0])

        out = self.outc(x_dec)
        out = self.out_act(out)
        out = self._unpad(out, pad)
        return out


class PerceptualLoss(nn.Module):
    """
    VGG16 feature-space MSE.
    Expects inputs in [-1, 1]; internally converts to ImageNet normalization.
    """
    def __init__(self, device=None, layers: int = 16):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Handle torchvision new weights API and older pretrained=True
        try:
            weights = models.VGG16_Weights.IMAGENET1K_FEATURES
            vgg = models.vgg16(weights=weights)
        except Exception:
            vgg = models.vgg16(pretrained=True)

        self.features = nn.Sequential(*list(vgg.features.children())[:layers]).to(device)
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imnet_mean", mean)
        self.register_buffer("imnet_std", std)

    def _to_imagenet(self, x):
        # x in [-1, 1] -> [0,1] -> ImageNet normalization
        x = (x + 1.0) / 2.0
        return (x - self.imnet_mean) / self.imnet_std

    def forward(self, pred, target):
        pred = self._to_imagenet(pred)
        target = self._to_imagenet(target)
        with torch.no_grad():
            self.features.eval()
        f_pred = self.features(pred)
        f_tgt = self.features(target)
        return F.mse_loss(f_pred, f_tgt)


class HybridLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, device=None):
        super().__init__()
        self.l1_w = float(l1_weight)
        self.perc_w = float(perceptual_weight)
        self.l1 = nn.L1Loss()
        self.perc = PerceptualLoss(device=device)

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        p = self.perc(pred, target)
        total = self.l1_w * l1 + self.perc_w * p
        return total, l1.item(), p.item()