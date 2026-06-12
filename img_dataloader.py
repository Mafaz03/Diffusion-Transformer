import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import io
from Scheduler import DDPM
import matplotlib.pyplot as plt

from torchvision import transforms


class dataset_imgs(Dataset):
    def __init__(self, folder: str,
                 betas_start: float = 1e-4,
                 betas_end: float = 1e-2,
                 max_timesteps: int = 1000):

        self.folder = Path(folder)
        allowed_exts = {".png", ".jpg", ".jpeg"}

        self.all_pths = [
            self.folder / name
            for name in os.listdir(self.folder)
            if Path(name).suffix.lower() in allowed_exts
        ]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

        self.ddpm = DDPM(betas_start, betas_end, max_timesteps)

        # 0..999
        self.ts = torch.arange(max_timesteps, dtype=torch.long)

    def __len__(self):
        return len(self.all_pths)

    def __getitem__(self, index):
        index = 0
        selected = self.all_pths[index]

        img = plt.imread(selected)

        number = float(selected.stem)

        img = self.transform(img)

        t = self.ts[torch.randint(0, len(self.ts), (1,))].item()

        x_t, noise = self.ddpm.add_noise(img.unsqueeze(0), torch.tensor([t], dtype=torch.long))

        x_t = x_t.squeeze(0)
        noise = noise.squeeze(0)

        return img, x_t, noise, t, number