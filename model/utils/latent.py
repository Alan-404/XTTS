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
    def __init__(self, embedding_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.num_samples_per_head = embedding_dim // num_heads
        self.scale = 1 / math.sqrt(self.num_samples_per_head)
        
        self.norm = GroupNorm32(embedding_dim)

        self.qkv_proj = nn.Conv1d(embedding_dim, embedding_dim * 3, kernel_size=1)
        self.out_proj = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

        # Init Weights
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def scaled_dot_product_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, qk_bias: int = 0):
        '''
            x: [batch_size, 3 * embedding_dim, length]
            - embedding_dim = n_samples_per_head * num_heads
        '''
        batch_size = x.size(0)

        q, k, v = x.reshape((batch_size, self.num_heads, 3 * self.num_samples_per_head, -1)).split(self.num_samples_per_head, dim=2) # (batch_size, num_heads, n_samples_per_head, length)
        q = q.transpose(-1, -2) # (batch_size, num_heads, length, n_samples_per_head)
        v = v.transpose(-1, -2) # (batch_size, num_heads, length, n_samples_per_head)

        q = q * self.scale
        k = k * self.scale

        score = torch.matmul(q, k) # [batch_size, num_heads, length, length]
        score = score + qk_bias
        if mask is not None:
            score.masked_fill_(mask, -torch.inf)
        
        weights = F.softmax(score.float(), dim=-1)
        context = torch.matmul(weights, v) # (batch_size, n_heads, length, n_samples_per_head)
        context = context.transpose(-1, -2).reshape((batch_size, self.embedding_dim, -1))

        return context

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, qk_bias: int = 0):
        batch_size = x.size(0)

        x = x.reshape((batch_size, self.embedding_dim, -1))
        x = self.norm(x)

        qkv = self.qkv_proj(x) # (batch_size, 3 * embedding_dim, length)
        h = self.scaled_dot_product_attention(qkv, mask, qk_bias)
        h = self.out_proj(h)
        
        out = x + h
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