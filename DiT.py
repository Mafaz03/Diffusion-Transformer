import torch
from patch_embedding import *

class DiT_Block(torch.nn.Module):
    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(d_model, elementwise_affine = False) # γ, β we will predict them overselfs for different t
        self.layer_norm_2 = torch.nn.LayerNorm(d_model, elementwise_affine = False) # γ, β we will predict them overselfs for different t

        self.mha = torch.nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, batch_first = True)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model * 4, d_model)
        )

        self.adaLN = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model * 6)
        ) 
        
        # adaLN-Zero: predict 6 modulation params from context
        # shift1, scale1  — for norm before attention
        # gate1           — gate on attention output
        # shift2, scale2  — for norm before MLP
        # gate2           — gate on MLP output

        # Zero-init the final linear so blocks start as identity
        torch.nn.init.zeros_(self.adaLN[-1].weight)
        torch.nn.init.zeros_(self.adaLN[-1].bias)

    
    def forward(self, patchified_inputs: torch.Tensor, context: torch.Tensor):
        # patchified_inputs: [B, seq_len, d_model]
        # context:           [B, d_model]; freq embedded + timestep embedded

        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(context).chunk(6, dim=-1)

        # attention block 
        patchified_inputs_norm = self.layer_norm_1(patchified_inputs)
        patchified_inputs_norm = patchified_inputs_norm * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        attn_outputs, _ = self.mha(patchified_inputs_norm, patchified_inputs_norm, patchified_inputs_norm) # self attention

        patchified_inputs = patchified_inputs + gate1.unsqueeze(1) * attn_outputs
        

        # mlp block 
        patchified_inputs_norm = self.layer_norm_2(patchified_inputs)
        patchified_inputs_norm = patchified_inputs_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        patchified_inputs = patchified_inputs + gate2.unsqueeze(1) * self.mlp(patchified_inputs_norm)

        return patchified_inputs


class DiT(torch.nn.Module):
    def __init__(self, d_model: int = 768, channels: int = 4, grid_size: int = 32, patch_size: int = 2, timestep_freq: int = 128,
                 num_freq: int = 128, num_DiT_blocks: int = 12, num_heads = 12):
        super().__init__()

        self.patchify = Patchify(channels = channels, patch_size = patch_size, d_model = d_model) # [B, C, H, W] -> [B, seq_len, d_model]
                                                                                                  # where seq_len = num_patch_x * num_patch_y
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = grid_size // patch_size

        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_patches * self.num_patches, d_model))
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        # self.register_buffer("position_embed", self.pos_embed)

        self.t_embed        = Timestep_Embedder(timestep_freq = timestep_freq, d_model = d_model)
        self.number_embed   = Fourier_Embedder(num_freqs = num_freq, d_model = d_model)

        self.blocks         = torch.nn.ModuleList([DiT_Block(d_model = d_model, num_heads = num_heads) for _ in range(num_DiT_blocks)])
        
        self.pixel_space_class = PixelSpace(d_model = d_model, channels = channels, patch_size = patch_size)

    def forward(self, noisy_latent: torch.Tensor, number: torch.Tensor, time: torch.Tensor):
        """
        noisy_latent: [B, C, H, W]  noisy latent
        time        : [B,]          diffusion timestep (int)
        number      : [B,]          conditioning number (float/int)
        """

        patchified_latents = self.patchify(grid = noisy_latent)                      # [B, C, H, W] -> [B, seq_len, d_model]
        patchified_latents = patchified_latents + self.pos_embed

        context = self.t_embed(t = time) + self.number_embed(number = number)        # [B, d_model]
        for block in self.blocks:
            patchified_latents = block(patchified_inputs = patchified_latents, 
                                       context = context)                            # [B, seq_len, d_model]
        
        pixel_space = self.pixel_space_class(patchified_input = patchified_latents,  # [B, seq_len, embed_dim] -> [B, seq_len, embed_dim']
                          context = context)

        unpatchified = Unpatchify(pixel_space = pixel_space, grid_size = self.grid_size, patch_size = self.patch_size, channels = 4) # [B, C, H, W]
        return unpatchified
    

if __name__ == "__main__":
    dit = DiT(d_model        = 768,
              channels       = 4,
              grid_size      = 32,
              patch_size     = 2,
              timestep_freq  = 128,
              num_freq       = 128,
              num_DiT_blocks = 12,
              num_heads      = 12)

    print(f"Total params: {sum(p.numel() for p in dit.parameters()):,}")

    batches    = 2
    channels   = 4
    grid_size  = 32

    noisy_latent = torch.rand(batches, channels, grid_size, grid_size)
    numbers      = torch.rand(batches)
    times        = torch.rand(batches)

    dit_out = dit(noisy_latent = noisy_latent, number = numbers, time = times)
    print(f"DiT output: {dit_out.shape}")