import torch


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

        patchified_inputs +=  gate1.unsqueeze(1) * attn_outputs

        # mlp block 
        patchified_inputs_norm = self.layer_norm_1(patchified_inputs)
        patchified_inputs_norm = patchified_inputs_norm * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)

        patchified_inputs +=  gate2.unsqueeze(1) * self.mlp(patchified_inputs_norm)

        return patchified_inputs
