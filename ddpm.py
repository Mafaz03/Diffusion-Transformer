import torch

class DDPM:
    """
    DDPM Scheduler
    

    # alphas = how much signal survives
    # betas  = how much noise is injected

    """

    def __init__(self, betas_start, betas_end, max_timesteps: int = 1000):
        self.betas = torch.linspace(betas_start, betas_end, max_timesteps) 

        self.alphas = 1. - self.betas

        self.alpha_bars_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alpha_bars_sqrt          = torch.sqrt(self.alpha_bars_cumprod)
        self._1_minus_alpha_bars_sqrt = torch.sqrt(1 - self.alpha_bars_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        # x0: [B, C, H, W]
        # t : [B, ]

        noise = torch.randn_like(x0)
        x_t = (self.alpha_bars_sqrt[t].view(-1, 1, 1, 1).to(x0.device) * x0) + (self._1_minus_alpha_bars_sqrt[t].view(-1, 1, 1, 1).to(x0.device) * noise)
        return x_t, noise
    
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    ddpm = DDPM(betas_start = 1e-4, betas_end = 1e-2, max_timesteps = 1000)

    x0 = cv2.cvtColor(cv2.resize(cv2.imread("sample_image.jpg"), (500,500)), cv2.COLOR_BGR2RGB)
    x0 = (torch.tensor(x0, dtype=torch.float32) / 255.0)

    x0 = torch.tensor(x0).permute(-1, 0, 1).unsqueeze(0)

    t = torch.asarray([10, 100, 200, 500, 600])

    noisy_x0, noises = ddpm.add_noise(x0, t)

    fix, axes = plt.subplots(1, 6, figsize = (30,30))

    axes[0].imshow(noisy_x0[0].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())
    axes[1].imshow(noisy_x0[1].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())
    axes[2].imshow(noisy_x0[2].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())
    axes[3].imshow(noisy_x0[3].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())
    axes[4].imshow(noisy_x0[4].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())
    axes[5].imshow(noises[0].permute(1,2,0).clamp(0, 1).detach().cpu().numpy())

    plt.show()