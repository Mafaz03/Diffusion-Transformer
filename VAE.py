import torch

class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.GroupNorm(32, channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1),
            torch.nn.GroupNorm(32, channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = 3, padding = 1),
        )
    
    def forward(self, x):
        return x + self.network(x)
    

class Encoder(torch.nn.Module):
    """
    Compresses (B, 3, H, W) -> (B, 8, H/8, W/8)
    Outputs 8 channels because we split into mu and logvar (4 each)
    for the reparameterization trick.
    """
    def __init__(self, ch=128, latent_channels=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            # Initial projection
            torch.nn.Conv2d(in_channels = 3, out_channels = ch, kernel_size = 3, padding = 1),

            # Downsample x2
            ResBlock(ch),
            torch.nn.Conv2d(in_channels = ch, out_channels = ch*2, kernel_size = 4, stride = 2, padding = 1),   # H/2

            # Downsample x4
            ResBlock(ch*2),
            torch.nn.Conv2d(in_channels = ch*2, out_channels = ch*4, kernel_size = 4, stride = 2, padding = 1), # H/4

            # Downsample x8
            ResBlock(ch*4),
            torch.nn.Conv2d(in_channels = ch*4, out_channels = ch*4, kernel_size = 4, stride = 2, padding = 1), # H/8

            ResBlock(ch*4),

            torch.nn.GroupNorm(32, ch*4),
            torch.nn.SiLU(),
            
            # Output mu and logvar concatenated on channel dim
            torch.nn.Conv2d(in_channels = ch*4, out_channels = latent_channels * 2, kernel_size = 1),
        )

    def forward(self, x):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=1)   # each [B, 4, H/8, W/8]
        return mu, logvar                
    



class Decoder(torch.nn.Module):
    """
    Reconstructs [B, 4, H/8, W/8] -> [B, 3, H, W]
    """

    def __init__(self, ch=128, latent_channels=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = latent_channels, out_channels = ch*4, kernel_size = 3, padding = 1),
            ResBlock(ch*4),

            # Upsample x4
            torch.nn.Upsample(scale_factor = 2, mode='nearest'),
            torch.nn.Conv2d(ch*4, ch*4, 3, padding=1),
            ResBlock(ch*4),

            # Upsample x2
            torch.nn.Upsample(scale_factor = 2, mode='nearest'),
            torch.nn.Conv2d(ch*4, ch*2, 3, padding=1),
            ResBlock(ch*2),

            # Upsample x1
            torch.nn.Upsample(scale_factor = 2, mode='nearest'),
            torch.nn.Conv2d(ch*2, ch, 3, padding=1),
            ResBlock(ch),

            torch.nn.GroupNorm(32, ch),
            torch.nn.SiLU(),
            torch.nn.Conv2d(ch, 3, 3, padding=1),
            torch.nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)
    


class VAE(torch.nn.Module):
    def __init__(self, ch = 128, latent_channels = 4):
        super().__init__()

        self.encoder = Encoder(ch = ch, latent_channels = latent_channels)
        self.decoder = Decoder(ch = ch, latent_channels = latent_channels)

    def encode(self, normal_grid):
        self.mu, self.logvar = self.encoder(normal_grid)
        return mu, logvar
    
    def decode(self, compressed_grid):
        reconstructed_grid = self.decoder(compressed_grid)
        return reconstructed_grid

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, grid):
        mu, logvar = self.encode(normal_grid = grid)        # each [B, 4, H/8, W/8]
        z = self.reparameterize(mu = mu, logvar = logvar)   # sampling from gausian distribution
        recon_grid = self.decode(compressed_grid = z)
        return recon_grid, mu, logvar
    

if __name__ == "__main__":
    img = torch.rand(3, 3, 256, 256)
    vae = VAE(ch = 128, latent_channels = 4)
    grid, mu, logvar = vae(img)
    print(grid.shape)
    print(mu.shape)
    print(logvar.shape)
