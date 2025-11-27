import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 256, base_channels: int = 32):
        super().__init__()
        
        # Encoder: 4 layers (Convolution -> BatchNorm -> LeakyReLU)
        self.encoder = nn.Sequential(
            # Layer 1: 3 -> base
            nn.Conv2d(3, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: base -> base*2
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: base*2 -> base*4
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: base*4 -> base*8
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(),
        )
        
        # Latent space statistics
        flat_features = base_channels * 8 * 16 * 16
        self.fc_mu = nn.Linear(flat_features, latent_dim)
        self.fc_logvar = nn.Linear(flat_features, latent_dim)
        
        # Decoder Input
        self.decoder_input = nn.Linear(latent_dim, flat_features)
        
        # Decoder: 4 layers (Upsampling)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_channels * 8, 16, 16)),
            
            # Layer 1: base*8 -> base*4
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: base*4 -> base*2
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: base*2 -> base
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: base -> 3
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1),
            nn.Tanh() # Output range [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        
        # Clamp logvar to prevent numerical instability (exploding KLD)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        z = self.reparameterize(mu, logvar)
        
        x_recon = self.decoder(self.decoder_input(z))
        
        return x_recon, mu, logvar