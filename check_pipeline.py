"""
check_pipeline.py  —  run this BEFORE training to verify your pipeline.

Usage:
    python check_pipeline.py --data_dir /path/to/images --vae_ckpt /path/to/vae.pt

All checks print PASS / FAIL / WARN with a short explanation.
Nothing is trained — the whole script runs in a few seconds.
"""

import argparse
import torch
from torch.utils.data import DataLoader

# ── helpers ────────────────────────────────────────────────────────────────────

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"

def _shape(t): return tuple(t.shape)

# ── checks ─────────────────────────────────────────────────────────────────────

def check_dataloader(data_dir: str):
    print("\n── Dataloader ──────────────────────────────────────────────────────")
    from img_dataloader import dataset_imgs
    ds = dataset_imgs(data_dir)
    img, x_t, noise, t, number = ds[0]

    print(f"{INFO} img    shape : {_shape(img)}")
    print(f"{INFO} x_t   shape : {_shape(x_t)}")
    print(f"{INFO} noise shape : {_shape(noise)}")
    print(f"{INFO} t           : {t}  (should be int in [0, 999])")
    print(f"{INFO} number      : {number}")

    C, H, W = img.shape
    if C == 3:
        print(f"{PASS} Image has 3 channels (RGB) — correct for VAE input.")
    else:
        print(f"{FAIL} Image has {C} channels — VAE probably expects 3.")

    if img.min() >= -1.1 and img.max() <= 1.1:
        print(f"{PASS} Image values in [-1, 1]  (min={img.min():.3f}, max={img.max():.3f})")
    else:
        print(f"{FAIL} Image values out of range (min={img.min():.3f}, max={img.max():.3f}) — check Normalize transform.")

    if _shape(x_t) == _shape(noise):
        print(f"{PASS} x_t and noise shapes match.")
    else:
        print(f"{FAIL} x_t {_shape(x_t)} and noise {_shape(noise)} shapes differ.")

    return C, H, W


def check_vae(vae_ckpt: str, img_C: int, img_H: int, img_W: int, device: str):
    print("\n── VAE latent shape ────────────────────────────────────────────────")
    try:
        from VAE import VAE
    except ImportError:
        print(f"{FAIL} Could not import VAE — skipping VAE checks.")
        return None, None

    # Build a dummy VAE; if you know your architecture args, pass them here.
    # We just load the checkpoint and infer the architecture from its weights.
    try:
        vae = VAE().to(device)
        if vae_ckpt:
            state = torch.load(vae_ckpt, map_location=device)
            vae.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state else state["state_dict"])
            print(f"{PASS} VAE checkpoint loaded from {vae_ckpt}")
    except Exception as e:
        print(f"{WARN} Could not load VAE checkpoint ({e}). Using random weights — shapes will still be correct.")
        vae = VAE().to(device)

    vae.eval()
    dummy_img = torch.zeros(1, img_C, img_H, img_W, device=device)

    with torch.no_grad():
        try:
            mu, logvar = vae.encode(dummy_img)
        except Exception as e:
            print(f"{FAIL} vae.encode() crashed: {e}")
            return None, None

    z_C, z_H, z_W = mu.shape[1], mu.shape[2], mu.shape[3]
    spatial_downsample = img_H // z_H

    print(f"{INFO} Input image  : {img_C} × {img_H} × {img_W}")
    print(f"{INFO} Latent mu    : {_shape(mu)}")
    print(f"{INFO} Latent channels  : {z_C}")
    print(f"{INFO} Spatial size     : {z_H} × {z_W}  (downsampled {spatial_downsample}×)")

    # Channel check
    if z_C == 4:
        print(f"{PASS} Latent has 4 channels — matches DiT(channels=4).")
    else:
        print(f"{FAIL} Latent has {z_C} channels but DiT expects channels=4.  "
              f"Fix: DiT(channels={z_C}) or change your VAE bottleneck.")

    # Spatial check
    if z_H == 32 and z_W == 32:
        print(f"{PASS} Latent is 32×32 — matches DiT(grid_size=32).")
    else:
        print(f"{FAIL} Latent is {z_H}×{z_W} but DiT expects grid_size=32.  "
              f"Fix: DiT(grid_size={z_H})  OR  resize input images to "
              f"{32 * spatial_downsample}×{32 * spatial_downsample}.")

    # Decode round-trip check
    with torch.no_grad():
        try:
            z = vae.reparameterize(mu, logvar)
            recon = vae.decode(z)
            if recon.shape == dummy_img.shape:
                print(f"{PASS} VAE decode round-trip shape OK: {_shape(recon)}")
            else:
                print(f"{FAIL} VAE decode output {_shape(recon)} != input {_shape(dummy_img)}")
        except Exception as e:
            print(f"{FAIL} vae.decode() crashed: {e}")

    return z_C, z_H


def check_dit_shapes(latent_C: int, latent_spatial: int, device: str):
    print("\n── DiT forward pass ────────────────────────────────────────────────")
    from DiT import DiT

    if latent_C is None or latent_spatial is None:
        print(f"{WARN} Skipping DiT check — VAE shapes unknown.")
        return

    dit = DiT(
        d_model        = 768,
        channels       = latent_C,       # match latent
        grid_size      = latent_spatial,  # match latent
        patch_size     = 2,
        timestep_freq  = 128,
        num_freq       = 128,
        num_DiT_blocks = 12,
        num_heads      = 12,
    ).to(device)

    B = 2
    dummy_latent = torch.randn(B, latent_C, latent_spatial, latent_spatial, device=device)
    dummy_t      = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)   # ← long ints, NOT rand()
    dummy_num    = torch.randn(B, device=device)

    try:
        with torch.no_grad():
            out = dit(noisy_latent=dummy_latent, time=dummy_t, number=dummy_num)

        if _shape(out) == _shape(dummy_latent):
            print(f"{PASS} DiT output shape {_shape(out)} matches input — pipeline compatible.")
        else:
            print(f"{FAIL} DiT output {_shape(out)} != latent {_shape(dummy_latent)}")

    except Exception as e:
        print(f"{FAIL} DiT forward pass crashed: {e}")
        return

    # Check timestep dtype — this is a common silent bug
    dummy_t_float = dummy_t.float()
    try:
        with torch.no_grad():
            out_float = dit(noisy_latent=dummy_latent, time=dummy_t_float, number=dummy_num)
        # If this doesn't crash, the embedder silently accepts floats — not necessarily correct
        print(f"{WARN} DiT accepted float timesteps without error.  "
              f"Verify Timestep_Embedder handles floats correctly — integers are expected.")
    except Exception:
        print(f"{PASS} DiT correctly rejects float timesteps (expected torch.long).")

    print(f"{INFO} Total DiT params: {sum(p.numel() for p in dit.parameters()):,}")


def check_scheduler(latent_C: int, latent_spatial: int, device: str):
    print("\n── Scheduler ───────────────────────────────────────────────────────")
    from Scheduler import DDPM

    sched = DDPM(betas_start=1e-4, betas_end=1e-2, max_timesteps=1000, device=device)

    if latent_C is None:
        print(f"{WARN} Skipping scheduler shape check — latent shape unknown.")
        return

    B = 2
    z   = torch.randn(B, latent_C, latent_spatial, latent_spatial, device=device)
    t   = torch.tensor([0, 999], device=device, dtype=torch.long)

    x_t, noise = sched.add_noise(z, t)

    if _shape(x_t) == _shape(z):
        print(f"{PASS} add_noise output shape matches input: {_shape(x_t)}")
    else:
        print(f"{FAIL} add_noise output {_shape(x_t)} != input {_shape(z)}")

    # Verify t=0 is almost clean, t=999 is almost pure noise
    snr_0   = sched.alpha_bars_sqrt[0].item()
    snr_999 = sched.alpha_bars_sqrt[999].item()
    print(f"{INFO} signal fraction at t=0  : {snr_0:.4f}  (should be ~1.0)")
    print(f"{INFO} signal fraction at t=999: {snr_999:.4f}  (should be ~0.0)")

    if snr_0 > 0.99:
        print(f"{PASS} t=0 is clean (signal fraction ≈ 1).")
    else:
        print(f"{WARN} t=0 signal fraction is {snr_0:.4f} — betas_start may be too large.")

    if snr_999 < 0.05:
        print(f"{PASS} t=999 is nearly pure noise (signal fraction ≈ 0).")
    else:
        print(f"{WARN} t=999 signal fraction is {snr_999:.4f} — betas_end may be too small for full noise.")

    # Verify remove_noise doesn't NaN
    noise_pred = torch.randn_like(x_t)
    for t_val in [999, 500, 1, 0]:
        out = sched.remove_noise(x_t, t_val, noise_pred)
        if torch.isnan(out).any():
            print(f"{FAIL} remove_noise produced NaN at t={t_val}")
        else:
            print(f"{PASS} remove_noise at t={t_val} — no NaNs.")


def check_latent_scale(vae_ckpt: str, data_dir: str, device: str):
    print("\n── Latent scale ────────────────────────────────────────────────────")
    try:
        from VAE import VAE
        from img_dataloader import dataset_imgs
        from training import compute_latent_scale
    except ImportError as e:
        print(f"{WARN} Skipping latent scale check ({e}).")
        return

    try:
        vae = VAE().to(device)
        if vae_ckpt:
            state = torch.load(vae_ckpt, map_location=device)
            vae.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state else state["state_dict"])
    except Exception as e:
        print(f"{WARN} Could not load VAE ({e}). Skipping latent scale check.")
        return

    ds = dataset_imgs(data_dir)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    scale = compute_latent_scale(vae, dl, device=device, n_batches=5)

    if 0.5 < scale < 5.0:
        print(f"{PASS} Latent scale = {scale:.4f} — looks reasonable.")
    else:
        print(f"{WARN} Latent scale = {scale:.4f} — unusually {'large' if scale >= 5 else 'small'}. "
              f"Check that your VAE is properly trained before computing this.")

    # Verify the scale direction
    print(f"{INFO} During training : z_scaled = z / {scale:.4f}  →  std ≈ 1")
    print(f"{INFO} During sampling : feed z_scaled to vae.decode(z_scaled * {scale:.4f})")
    print(f"{WARN} Double-check sample_from_dit ends with: vae.decode(x * latent_scale)  NOT  x / latent_scale")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True,  help="Path to your image folder")
    parser.add_argument("--vae_ckpt",  default="",     help="Path to saved VAE checkpoint (optional)")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  DiT pipeline diagnostic")
    print(f"  device: {args.device}")
    print(f"{'='*60}")

    img_C, img_H, img_W = check_dataloader(args.data_dir)
    latent_C, latent_spatial = check_vae(args.vae_ckpt, img_C, img_H, img_W, args.device)
    check_dit_shapes(latent_C, latent_spatial, args.device)
    check_scheduler(latent_C, latent_spatial, args.device)
    check_latent_scale(args.vae_ckpt, args.data_dir, args.device)

    print(f"\n{'='*60}")
    print("  Done. Fix all FAIL lines before starting DiT training.")
    print(f"{'='*60}\n")
