import torch

class DDPM:
    """
    DDPM Scheduler
    

    # alphas = how much signal survives
    # betas  = how much noise is injected

    """

    def __init__(self, betas_start, betas_end, max_timesteps: int = 1000, device:str = "cpu"):
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

    

    def remove_noise(self, xt, t, noise):
        # DDPM reverse step: x_t -> x_{t-1}

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=xt.device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.unsqueeze(0)

        alpha_t     = self.alphas[t].view(-1, 1, 1, 1).to(xt.device)
        beta_t      = self.betas[t].view(-1, 1, 1, 1).to(xt.device)
        alpha_bar_t = self.alpha_bars_cumprod[t].view(-1, 1, 1, 1).to(xt.device)

        mean = (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise) / torch.sqrt(alpha_t)

        if t[0].item() == 0:
            return mean

        # posterior variance
        alpha_bar_prev = self.alpha_bars_cumprod[t - 1].view(-1, 1, 1, 1).to(xt.device)
        beta_tilde_t   = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

        z = torch.randn_like(xt)
        return mean + torch.sqrt(beta_tilde_t) * z
    
    def remove_noise(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        # DDPM reverse step: x_t -> x_{t-1}
        
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=xt.device, dtype=torch.long)
        elif t.ndim == 0:
            t = t.unsqueeze(0)

        alpha_t     = self.alphas[t].view(-1, 1, 1, 1).to(xt.device)
        beta_t      = self.betas[t].view(-1, 1, 1, 1).to(xt.device)
        alpha_bar_t = self.alpha_bars_cumprod[t].view(-1, 1, 1, 1).to(xt.device)

        mean = (xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise) / torch.sqrt(alpha_t)

        if t[0].item() == 0:
            return mean

        alpha_bar_prev = self.alpha_bars_cumprod[t - 1].view(-1, 1, 1, 1).to(xt.device)
        beta_tilde_t   = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t  # ← correct posterior variance

        z = torch.randn_like(xt)
        return mean + torch.sqrt(beta_tilde_t) * z