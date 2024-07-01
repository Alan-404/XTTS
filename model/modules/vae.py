import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed

from typing import Callable

class VectorQuantization(nn.Module):
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5, balancing_heristic: bool = False, new_return_order: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.balancing_heuristic = balancing_heristic
        self.new_return_order = new_return_order

        self.codes = None
        self.max_codes = 64000
        self.codes_full = False

        embed = torch.randn((dim, n_embed))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x: torch.Tensor, return_soft_codes: bool = False):
        batch_size, length, _ = x.size()

        if self.balancing_heuristic and self.codes_full:
            h = torch.histc(self.codes, bins=self.n_embed, min=0, max=self.n_embed) / len(self.codes)
            mask = torch.logical_or(h > 0.9, h < 0.01).unsqueeze(1)
            ep = self.embed.permute(1, 0)
            ea = self.embed_avg.permute(1, 0)
            rand_embed = torch.randn_like(ep) * mask
            self.embed = (ep * (~mask) + rand_embed).permute(1, 0)
            self.embed_avg = (ea * (~mask) + rand_embed).permute(1, 0)
            self.cluster_size = self.cluster_size * (~mask).squeeze()
            if torch.any(mask):
                self.codes = None
                self.codes_full = False
            
        flatten = x.reshape((batch_size * length, self.dim))
        dist = x.pow(2).sum(dim=1, keepdim=True) - 2 * torch.matmul(flatten, self.embed) + self.embed.pow(2).sum(dim=1, keepdim=True) # (batch_size * length, dim)
        soft_codes = -dist
        _, embed_ind = torch.max(soft_codes, dim=1) # (batch_size * length)
        quantize = self.embed_code(embed_ind.reshape((batch_size, length)))

        if self.balancing_heuristic:
            if self.codes is None:
                self.codes = embed_ind # (batch_size, length)
            else:
                self.codes = torch.cat([self.codes, embed_ind])
                if len(self.codes) > self.max_codes:
                    self.codes = self.codes[-self.max_codes:]
                    self.codes_full = True

        if self.training:
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(x.dtype) # (batch_size * length, n_embed)

            embed_onehot_sum = embed_onehot.sum(dim=0) # (n_embed)
            embed_sum = torch.matmul(flatten.transpose(0, 1), embed_onehot)

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)
            
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        embed_ind = embed_ind.reshape((batch_size, length))

        if return_soft_codes:
            return quantize, diff, embed_ind, soft_codes.reshape((batch_size, length, self.dim))
        elif self.new_return_order:
            return quantize, embed_ind, diff
        else:
            return quantize, diff, embed_ind

    def embed_code(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.embed.transpose(0, 1))
    
class ResBlock(nn.Module):
    def __init__(self, channels: int, conv: nn.Module, activation: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv(channels, channels, 3, padding=1),
            activation(),
            conv(channels, channels, 3, padding=1),
            activation(),
            conv(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x) + x
        return x
    
class UpsampleConv(nn.Module):
    def __init__(self, conv: nn.Module, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = conv()
            
class VQVAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)