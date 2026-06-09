import torch
import torch.nn as nn
import math


class Patchify(torch.nn.Module):
    def __init__(self, in_channels: int = 4, patch_size: int = 2, d_model: int = 768):
        super().__init__()
        
        # [B, C, H, W] -> [B, seq_len, d_model]
        # where seq_len = num_patch_x * num_patch_y

        self.patch_size = patch_size

        self.d_model_expander = torch.nn.Linear(in_channels * self.patch_size * self.patch_size, d_model)


    def forward(self, grid: torch.Tensor):
        # grid: [B, C, H, W]; post VAE 

        B, C, H, W = grid.shape
        
        grid = grid.reshape(B, C * self.patch_size * self.patch_size, H//self.patch_size, W//self.patch_size) # [B, embed_dim', num_patches_x, num_patches_y]
        grid = grid.reshape(B, C * self.patch_size * self.patch_size, -1)                                     # [B, embed_dim', num_patches_x * num_patches_y] basically [B, embed_dim', seq_len]
        grid = grid.transpose(2, 1)                                                                           # [B, seq_len, embed_dim']
        return self.d_model_expander(grid)                                                                    # [B, seq_len, embed_dim]
    



class Unpatchify(torch.nn.Module):
    """
    Back to pixel space from d_model dimension
    """
    def __init__(self, d_model: int = 768, channels: int = 4, patch_size: int = 2):
        super().__init__()

        self.norm   = nn.LayerNorm(d_model, elementwise_affine=False)

        self.adaLN = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, 2 * d_model)
        )

        self.patch_size = patch_size
        self.d_model_shrinker = torch.nn.Linear(d_model, channels * self.patch_size * self.patch_size)       # [B, seq_len, embed_dim] -> [B, seq_len, embed_dim']


        torch.nn.init.zeros_(self.adaLN[-1].weight)
        torch.nn.init.zeros_(self.adaLN[-1].bias)
        torch.nn.init.zeros_(self.d_model_shrinker.weight)
        torch.nn.init.zeros_(self.d_model_shrinker.bias)


    def forward(self, patchified_input: torch.Tensor, context: torch.Tensor):
        shift, scale = self.adaLN(context).chunk(2, dim = -1)
        
        patchified_input = (self.norm(patchified_input) * (1 + scale.unsqueeze(1))) + shift.unsqueeze(1)
        return self.d_model_shrinker(patchified_input)




class Timestep_Embedder(torch.nn.Module):
    def __init__(self, timestep_freq: int, d_model: int):
        super().__init__()

        self.timestep_freq = timestep_freq

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(timestep_freq, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model)
        )

    def get_timestep_embedding(self, t: torch.Tensor, dim: int):
        """
        Sinusoidal timestep embedding 
        """

        # t: [B,] -> [B, timestep_freq]

        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1)) # [half]
        args  = t.unsqueeze(-1).float() * freqs.unsqueeze(0)                                   # [B, half]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)                          # [B, half * 2] -> [B, timestep_freq]
        return emb
    
    def forward(self, t: torch.Tensor):
        emb = self.get_timestep_embedding(t = t, dim = self.timestep_freq)
        return self.mlp(emb)



class Fourier_Embedder(torch.nn.Module):
    """
    Single number input -> embedding
    """

    def __init__(self, num_freqs = 128, d_model: int = 768):
        super().__init__()

        self.register_buffer("freqs", torch.randn(num_freqs) * 10)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * num_freqs, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model)
        )

    def forward(self, number):
        # number: [B, ]
        x = number.unsqueeze(1) * self.freqs.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim = -1) # [B, 2 * num_freqs]
        return self.mlp(x)                                          # [B, d_model]
    


if __name__ == "__main__":
    patch_class   = Patchify(channels = 4, patch_size = 2)
    unpatch_class = Unpatchify(d_model = 768, channels = 4, patch_size = 2)

    post_vae = torch.rand(3, 4, 32, 32)

    patchified = patch_class(post_vae)
    print(f"Patchified shape: {patchified.shape}")

    unpatchified = unpatch_class(patchified, torch.rand(patchified.shape[0], patchified.shape[-1]))
    print(f"Un-patchified shape: {unpatchified.shape}")
    

    timestep_embedder = Timestep_Embedder(timestep_freq = 256, d_model = 768)
    t = torch.randint(0, 10, (10,))

    print(timestep_embedder(t).shape)


