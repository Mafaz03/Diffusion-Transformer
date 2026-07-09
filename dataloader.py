import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from torchvision import transforms
from Scheduler import DDPM

from PIL import Image
import os


class dataset(Dataset):
    """
    `number` is just a dummy 0.0 placeholder. This keeps the return
    signature identical to dataset_imgs (img, x_t, noise, t, number) so
    it's a drop-in replacement in training loop.
    """

    def __init__(self,
                 split: str = "train",
                 betas_start: float = 1e-4,
                 betas_end: float = 0.02,
                 max_timesteps: int = 1000,
                 hf_dataset_name: str = "korexyz/celeba-hq-256x256",
                 dataset_root: str = None):


        
        self.dataset_root = dataset_root
        if dataset_root is not None:
            self.image_paths = [
                os.path.join(dataset_root, f)
                for f in os.listdir(dataset_root)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]

        else:
            self.ds = load_dataset(hf_dataset_name, split=split)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

        self.ddpm = DDPM(betas_start, betas_end, max_timesteps)
        self.ts = torch.arange(max_timesteps, dtype=torch.long)

    def __len__(self):
        if self.dataset_root is not None:
            return len(self.image_paths)

        return len(self.ds)

    def __getitem__(self, index):
        

        if self.dataset_root is not None:
            img = Image.open(self.image_paths[index]).convert("RGB")
        else:
            img = self.ds[index]["image"].convert("RGB")          # PIL Image

        img = self.transform(img)
        # import pdb; pdb.set_trace()
        # number = torch.tensor(float(self.image_paths[index].split("/")[-1].split(".jpg")[0]))           # TODO: fix for .jpg/.png/.exr etc
        filename = os.path.basename(self.image_paths[index])   # 123.png
        stem = os.path.splitext(filename)[0]                   # 123
        number = torch.tensor(float(stem))

        t = self.ts[torch.randint(0, len(self.ts), (1,))].item()

        x_t, noise = self.ddpm.add_noise(img.unsqueeze(0), torch.tensor([t], dtype=torch.long))

        x_t = x_t.squeeze(0)
        noise = noise.squeeze(0)

        return img, x_t, noise, t, number


if __name__ == "__main__":
 
    dataset = dataset(split="train", dataset_root = "LidDriven_imgs")
    print(f"Dataset size: {len(dataset)}")

    img, x_t, noise, t, number = dataset[0]
    print(f"img:    {img.shape}  range[{img.min():.2f}, {img.max():.2f}]")
    print(f"x_t:    {x_t.shape}")
    print(f"noise:  {noise.shape}")
    print(f"t:      {t}")
    print(f"number: {number}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)#, num_workers=2) # some macos problem
    batch = next(iter(loader))
    imgs, x_ts, noises, ts, numbers = batch
    print(f"\nBatch shapes: imgs={imgs.shape}  numbers={numbers.shape}")

    numbers_list = torch.tensor([], dtype=torch.float32)
    for imgs, x_ts, noises, ts, numbers in loader:
        numbers_list = torch.cat((numbers_list, numbers), dim=0)

    print("Mean: ", numbers_list.mean().item())
    print("Std: ", numbers_list.std().item())

