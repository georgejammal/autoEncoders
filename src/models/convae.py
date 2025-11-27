import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1 ),nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1 ),nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x