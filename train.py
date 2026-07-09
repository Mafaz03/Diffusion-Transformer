import torch
import matplotlib.pyplot as plt
import cv2

from VAE import VAE, vae_loss
from training_v2 import *
from training import train_vae

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from img_dataloader import dataset_imgs

import numpy as np

import json

from DiT_v2 import *
from DDPM import *




########################################
########## Loading modules #############
########################################

print("Loading modules.....")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device found: {device}")



with open("config.json", "r") as file:
    config = json.load(file)

scheduler = LinearNoiseScheduler(num_timesteps  = config["Scheduler"]["num_timesteps"],
                                     beta_start = config["Scheduler"]["beta_start"],
                                     beta_end   = config["Scheduler"]["beta_end"])




# vae = VAE(ch = 128, latent_channels = 4).to(device)
vae = VAE(device = device, freeze = True, scaling_factor = config["VAE"]["scaling_factor"], path = "sd-vae-ft-mse").to(device)


dit = DiT(d_model           = config["DiT"]["d_model"],
          g_channels        = config["DiT"]["g_channels"],
          grid_size         = config["DiT"]["grid_size"],
          patch_size        = config["DiT"]["patch_size"],
          timestep_emb_dim  = config["DiT"]["timestep_emb_dim"],
          number_emb_dim    = config["DiT"]["number_emb_dim"],
          num_layers        = config["DiT"]["num_layers"],
          num_heads         = config["DiT"]["num_heads"])


dit = dit.to(device)
vae = vae.to(device)

# dit.load_state_dict(torch.load('DiT_landscape.pth', map_location = device))

########################################
########### Training DiT  ##############
########################################

print("Training DiT ......")


from dataloader import dataset

# dataset    = dataset(split="train", dataset_root = "celeba_hq_256")
dataset    = dataset(split="train", dataset_root = "celeba_hq_256")
dataloader = DataLoader(dataset, batch_size = config["Training"]["batch_size"], shuffle=True, num_workers=2)

dit_losses = train(config["Training"]["epochs"], dataloader, dit, vae, scheduler, device, acc_steps = config["Training"]["accumulation_step"])