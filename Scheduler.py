import torch

class DDPM:
    """
    DDPM Scheduler
    

    # alphas = how much signal survives
    # betas  = how much noise is injected

    """

    def __init__(self, betas_start, betas_end, max_timesteps: int = 1000, device:str = "cuda"):
        self.max_timesteps = max_timesteps
        self.betas = torch.linspace(betas_start, betas_end, max_timesteps).to(device) 

        self.alphas = 1. - self.betas

        self.alpha_bars_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        self.alpha_bars_sqrt          = torch.sqrt(self.alpha_bars_cumprod).to(device)
        self._1_minus_alpha_bars_sqrt = torch.sqrt(1 - self.alpha_bars_cumprod).to(device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        # x0: [B, C, H, W]
        # t : [B, ]

        noise = torch.randn_like(x0)
        x_t = (self.alpha_bars_sqrt[t].view(-1, 1, 1, 1).to(x0.device) * x0) + (self._1_minus_alpha_bars_sqrt[t].view(-1, 1, 1, 1).to(x0.device) * noise)
        return x_t, noise

    def remove_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        # x0: [B, C, H, W]
        # t : [B, ]

        alpha_bar_sqrt           = self.alpha_bars_sqrt[t].view(-1,1,1,1).to(xt.device)
        one_minus_alpha_bar_sqrt = self._1_minus_alpha_bars_sqrt[t].view(-1,1,1,1).to(xt.device)


        return (xt - one_minus_alpha_bar_sqrt * noise) / alpha_bar_sqrt