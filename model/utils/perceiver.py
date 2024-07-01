import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.attention import MultiHeadAttention

import math

from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True) -> None:
        super().__init__()
        self.scale = math.sqrt(dim)

        self.gamma = nn.Parameter(torch.ones(dim)) if scale else 1

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, dim=-1) * self.scale * self.gamma
        return x
    
class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dialation: int = 1) -> None:
        super().__init__()
        self.causal_padding = dialation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = self.conv(x)
        return x

class GEGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        x = F.gelu(gate) * x
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, causal_conv: bool = False) -> None:
        super().__init__()
        dim_inner = int(dim * mult * 2 / 3)
        self.causal_conv = causal_conv

        if causal_conv:
            self.conv = CausalConv1d(dim_inner ,dim_inner, kernel_size=3)

        self.hidden_linear = nn.Linear(dim, dim_inner * 2)
        self.activation = GEGLU()
        self.out_linear = nn.Linear(dim_inner, dim)

    def forward(self, x: torch.Tensor):
        x = self.hidden_linear(x)
        x = self.activation(x)
        if self.causal_conv:
            x = x.transpose(-1, -2)
            x = self.conv(x)
            x = x.transpose(-1, -2)
        x = self.out_linear(x)
        return x

class PerceiverResampler(nn.Module):
    def __init__(self, in_dim: int, dim: int, depth: int = 2, num_latents: int = 32, n_heads: int = 8, ff_mult: int = 4) -> None:
        super().__init__()
        self.proj_context = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    MultiHeadAttention(
                        dim=dim,
                        n_heads=n_heads
                    ),
                    FeedForward(dim, ff_mult)
                ])
            )

        self.norm = RMSNorm(dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.proj_context(x)
        latents = self.latents.repeat([batch_size, 1, 1])
        for attn, ffn in self.layers:
            latents = attn(latents, x, x, mask=mask) + latents
            latents = ffn(latents) + latents
        
        latents = self.norm(latents)
        return latents