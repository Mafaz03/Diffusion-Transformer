import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from DiT_v2 import DiT
from VAE import VAE
from DDPM import LinearNoiseScheduler
from img_dataloader import dataset_imgs

from torchvision.utils import make_grid
import torchvision


# num_timesteps = 1000
# beta_start    = 0.0001
# beta_end      = 0.02


# scheduler = LinearNoiseScheduler(num_timesteps  = num_timesteps,
#                                      beta_start = beta_start,
#                                      beta_end   = beta_end)


# device = "cuda" if torch.cuda.is_available() else "cpu"

# vae = VAE(ch = 128, latent_channels = 4).to(device)


# dit = DiT(d_model        = 256,
#           g_channels       = 4,
#           grid_size      = 32,
#           patch_size     = 4,
#           timestep_emb_dim  = 128,
#         #   num_freq       = 128,
#           num_layers     = 8,
#           num_heads      = 4)

# vae.eval()
# dit.train()


def train(epochs, dataloader, dit, vae, scheduler, device):
    optimizer = AdamW(dit.parameters(), lr=1E-5, weight_decay=0)
    loss_fn   = torch.nn.MSELoss()

    for param in vae.parameters():
        param.requires_grad = False

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, _x_t, _noise, _t, numbers in dataloader:
            images  = images.to(device)
            # numbers = numbers.float().to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar)

            # Sample random noise
            noise = torch.randn_like(z).to(device)

            # Sample timestep
            t = torch.randint(0, 1000,(z.shape[0],)).to(device)

            noisy_im = scheduler.add_noise(z, noise, t)

            pred = dit(noisy_im, t)
            loss = loss_fn(pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
    return losses
