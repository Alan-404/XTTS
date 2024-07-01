import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout

        self.n_samples_per_head = dim // n_heads
        self.sqrt_dim = math.sqrt(self.n_samples_per_head)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attention_socres = torch.matmul(q, k.transpose(-1, -2))
        attention_socres = attention_socres / self.sqrt_dim

        if mask is not None:
            attention_socres.masked_fill_(mask, float('-inf'))
        attention_weights = F.softmax(attention_socres, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)

        attention_context = torch.matmul(attention_weights, v)
        return attention_context
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, query_length, _ = q.size()
        cross_length = k.size(1)

        # Projection
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Split Heads
        q = q.reshape((batch_size, query_length, self.n_heads, self.n_samples_per_head)).transpose(1, 2)
        k = k.reshape((batch_size, cross_length, self.n_heads, self.n_samples_per_head)).transpose(1, 2)
        v = v.reshape((batch_size, cross_length, self.n_heads, self.n_samples_per_head)).transpose(1, 2)

        # Attention
        attention_context = self.scaled_dot_product_attention(q, k, v, mask)
        attention_context = attention_context.transpose(1, 2).reshape((batch_size, query_length, self.dim))
        attention_context = self.out_proj(attention_context)

        return attention_context