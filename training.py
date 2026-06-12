
import torch
import matplotlib.pyplot as plt
import cv2

from VAE import VAE, vae_loss

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from img_dataloader import dataset_imgs

from DiT import DiT
from Scheduler import DDPM


def compute_latent_scale(vae: VAE, dataloader: DataLoader, device: str, n_batches: int = 50) -> float:
    """
    Compute the empirical std of the VAE latent space so DiT trains
    on unit-variance latents.
 
    latent_scale  ->  z_scaled = z / latent_scale  (std approx 1)
    At sampling   ->  z        = z_scaled * latent_scale  before decode
    """
    vae.eval()
    all_stds = []
    with torch.no_grad():
        for i, (images, *_ ) in enumerate(dataloader):
            if i >= n_batches:
                break
            images = images.to(device)
            mu, logvar = vae.encode(images)
            z = vae.reparameterize(mu, logvar)
            all_stds.append(z.std().item())
    scale = float(torch.tensor(all_stds).mean())
    print(f"[latent_scale] empirical latent std = {scale:.4f}")
    return scale


def train_vae(vae, dataloader, epochs=100, device = "cuda"):
    vae.train()
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


def train_dit(
    model:        DiT,
    vae:          VAE,
    dataloader:   DataLoader,
    scheduler:    DDPM,
    latent_scale: float,
    epochs:       int   = 10,
    lr:           float = 1e-4,
    device:       str   = "cuda",
):
    model.to(device)
    vae.to(device)
 
    
    scheduler.betas                  = scheduler.betas.to(device)
    scheduler.alphas                 = scheduler.alphas.to(device)
    scheduler.alpha_bars_cumprod     = scheduler.alpha_bars_cumprod.to(device)
    scheduler.alpha_bars_sqrt        = scheduler.alpha_bars_sqrt.to(device)
    scheduler._1_minus_alpha_bars_sqrt = scheduler._1_minus_alpha_bars_sqrt.to(device)
 
    # Freeze VAE
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses    = []
 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
 
        for images, _x_t, _noise, _t, numbers in dataloader:
            images  = images.to(device)
            numbers = numbers.float().to(device)
 
            with torch.no_grad():
                mu, logvar = vae.encode(images)
                # z = vae.reparameterize(mu, logvar) / latent_scale  # unit variance
                z = mu / latent_scale  # unit variance
 
            B = z.shape[0]
            t = torch.randint(0, scheduler.max_timesteps, (B,), device=device, dtype=torch.long)
 
            x_t, noise = scheduler.add_noise(z, t)
 
            noise_pred = model(noisy_latent=x_t, time=t, number=numbers)
 
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
            epoch_loss += loss.item()
 
        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"[DiT] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
 
    return losses

def train_joint(
    model:        DiT,
    vae:          VAE,
    dataloader:   DataLoader,
    scheduler:    DDPM,
    latent_scale: float,
    epochs:       int   = 5,
    lr_dit:       float = 1e-4,
    lr_vae:       float = 1e-5,   # 10× smaller – VAE is already converged
    kl_weight:    float = 1e-4,
    device:       str   = "cuda",
):
    """
    Joint fine-tuning of VAE + DiT.
 
    Loss = diffusion_loss  +  vae_reconstruction_loss  +  kl_weight * kl_loss
 
    The diffusion loss gradient flows through latent z back into the VAE
    encoder because we no longer use torch.no_grad() around the encode step.
    """
    model.to(device)
    vae.to(device)
 
    scheduler.betas                    = scheduler.betas.to(device)
    scheduler.alphas                   = scheduler.alphas.to(device)
    scheduler.alpha_bars_cumprod       = scheduler.alpha_bars_cumprod.to(device)
    scheduler.alpha_bars_sqrt          = scheduler.alpha_bars_sqrt.to(device)
    scheduler._1_minus_alpha_bars_sqrt = scheduler._1_minus_alpha_bars_sqrt.to(device)
 
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr_dit},
        {"params": vae.parameters(),   "lr": lr_vae},
    ])
 
    losses = []
 
    for epoch in range(epochs):
        model.train()
        vae.train()
        epoch_loss = 0.0
 
        for images, _x_t, _noise, _t, numbers in dataloader:
            images  = images.to(device)
            numbers = numbers.float().to(device)
 
            # VAE forward
            mu, logvar = vae.encode(images)
            z          = vae.reparameterize(mu, logvar) / latent_scale
            recon      = vae.decode(z * latent_scale)
 
            vae_l = vae_loss(recon, images, mu, logvar, kl_weight=kl_weight)
 
            # Diffusion forward 
            B = z.shape[0]
            t = torch.randint(0, scheduler.max_timesteps, (B,), device=device, dtype=torch.long)
 
            x_t, noise = scheduler.add_noise(z, t)
 
            noise_pred = model(noisy_latent=x_t, time=t, number=numbers)
            diff_loss  = torch.nn.functional.mse_loss(noise_pred, noise)
 
            loss = diff_loss + vae_l
 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(vae.parameters(),   1.0)
            optimizer.step()
 
            epoch_loss += loss.item()
 
        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        print(f"[Joint] Epoch {epoch+1}/{epochs}  loss={avg:.6f}")
 
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
        t_batch    = torch.tensor([t], device=device, dtype=torch.long)
        noise_pred = model(noisy_latent = x, time = t_batch, number = n)
        
        x = scheduler.remove_noise(xt    = x, 
                                   t     = t, 
                                   noise = noise_pred)
        if torch.isnan(x).any():
            print("NaN at timestep:", t)
            break

    # Decode latent -> image
    image = vae.decode(x * latent_scale)
    return image