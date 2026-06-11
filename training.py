
import torch
import matplotlib.pyplot as plt
import cv2

from VAE import VAE, vae_loss

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from img_dataloader import dataset_imgs

from DiT import DiT
from Scheduler import DDPM




def train_vae(vae, dataloader, epochs=100, device = "cuda"):
    losses = []
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)

    step = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for images, x_t, noise, t, number in dataloader:
            images = images.to(device)
            recon, mu, logvar = vae(images)

            # Anneal KL weight from 0 -> 1e-4 over first 10k steps
            # to avoid posterior collapse early in training
            kl_w = min(1e-4, step / 10_000 * 1e-4)
            loss = vae_loss(recon, images, mu, logvar, kl_weight=kl_w)
            epoch_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
        
        losses.append(epoch_loss/len(dataloader))
        print(f"Epoch: {epoch} / {epochs} => loss: {epoch_loss/len(dataloader):.5f}")
    return losses


def train_dit(model: DiT, vae: VAE, dataloader, scheduler: DDPM, latent_scale: float, epochs = 10, lr=1e-4, device="cuda", freeze_VAE: bool = True):
    model.to(device)
    vae.to(device)

    # Freeze VAE
    if freeze_VAE:
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr
    )

    losses = []

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0

        for images, _, _, _, numbers in dataloader:

            images = images.to(device)
            numbers = numbers.float().to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(images)
                z = vae.reparameterize(mu, logvar) * latent_scale

            B = z.shape[0]

            t = torch.randint(0, scheduler.max_timesteps, (B,),device=device)

            x_t, noise = scheduler.add_noise(z, t)

            noise_pred = model(x_t, t.float(), numbers)

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {avg_loss:.6f}"
        )

    return losses


@torch.no_grad()
def sample_from_dit(model, vae: VAE, n_value, scheduler: DDPM, latent_scale: float, img_size = 256, device='cuda'):
    """Generate an image conditioned on a specific number."""

    model.eval()
    vae.eval()

    # Start from pure noise
    x = torch.randn(1, 4, img_size // 8, img_size // 8, device=device)
    n = torch.tensor([n_value], dtype=torch.float32, device=device)

    for t in tqdm(reversed(range(scheduler.max_timesteps)), total=scheduler.max_timesteps):
        t_batch = torch.tensor([t], device=device)
        noise_pred = model(noisy_latent = x, time = t_batch, number = n)
        
        x = scheduler.remove_noise(xt    = x, 
                                   t     = t, 
                                   noise = noise_pred)
        if torch.isnan(x).any():
            print("NaN at timestep:", t)
            break

    # Decode latent -> image
    image = vae.decode(x / latent_scale)
    return image