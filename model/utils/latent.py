import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class GroupNorm32(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 32
        if channels <= 16:
            groups = 8
        elif channels <= 64:
            groups = 16
        self.layer = nn.GroupNorm(groups, channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.float()).type(x.dtype)
    
class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1, out_channels: Optional[int] = None, do_activation: bool = False) -> None:
        super().__init__()
        self.channels = channels
        if out_channels is None:
            out_channels = channels

        self.do_activation = do_activation
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads = num_heads // num_head_channels
        
        self.norm = GroupNorm32(channels=channels)
        self.qkv = nn.Conv1d(channels, out_channels*3, kernel_size=1)

        self.x_proj = nn.Identity() if out_channels == channels else nn.Conv1d(channels, out_channels, kernel_size=1)
        self.proj_out = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # Init Weights
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def scaled_dot_product_attention(self, qkv: torch.Tensor, mask: Optional[torch.Tensor], qk_bias: int = 0):
        batch_size, width, length = qkv.size()

        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape((batch_size * self.num_heads, ch * 3, length)).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        q = q * scale
        k = k * scale

        score = torch.matmul(q, k.transpose(-1, -2))
        score = score + qk_bias
        if mask is not None:
            score.masked_fill_(mask, -torch.inf)
        
        weights = F.softmax(score.float(), dim=-1)
        context = torch.matmul(weights, v)
        context = context.reshape(batch_size, -1, length)

        return context

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, qk_bias: int = 0):
        batch_size, channels, *spatial = x.size()

        x = x.reshape((batch_size, channels, -1))
        x = self.norm(x)
        if self.do_activation:
            x = F.silu(x, inplace=True)
        qkv = self.qkv(x)
        h = self.scaled_dot_product_attention(qkv, mask, qk_bias)
        h = self.proj_out(h)

        xp = self.x_proj(x)
        
        out = (xp + h).reshape((batch_size, xp.size(1), *spatial))
        return out

class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim: int, embedding_dim: int, attn_blocks: int = 6, num_attn_heads: int = 4) -> None:
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.ModuleList()
        for _ in range(attn_blocks):
            self.attn.append(AttentionBlock(embedding_dim, num_attn_heads))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init(x)
        for layer in self.attn:
            x = layer(x)
        return x
        