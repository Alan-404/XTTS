import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as distributed

class VectorQuantization(nn.Module):
    def __init__(self, dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5, balancing_heuristic: bool = False, new_return_order: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.balancing_heuristic = balancing_heuristic
        self.new_retun_order = new_return_order

        self.codes = None
        self.codes_full = False
        self.max_codes = 64000
        embed = torch.randn((dim, n_embed))
        
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def embed_code(self, x: torch.Tensor) -> torch.Tensor:
        x = F.embedding(x, self.embed.transpose(0, 1))
        return x
    
    def forward(self, x: torch.Tensor, return_soft_codes: bool = False):
        batch_size, length, _ = x.size()

        if self.balancing_heuristic and self.codes_full:
            h = torch.histc(self.codes, bins=self.n_embed, min=0, max=self.n_embed)
            mask = torch.logical_or(h>0.9, h<0.01).unsqueeze(1)
            ep = self.embed.permute(1, 0)
            ea = self.embed_avg.permute(1, 0)
            rand_embed = torch.randn_like(ep) * mask

            self.embed = (ep * (~mask) + rand_embed).permute(1, 0)
            self.embed_avg = (ea * (~mask) + rand_embed).permute(1, 0)
            self.cluster_size = self.cluster_size * (~mask).squeeze()
            if torch.any(mask):
                self.codes = None
                self.codes_full = False
        
        flatten = x.reshape((batch_size * length, self.dim)) # shape = [batch_size * length, dim]
        dist = flatten.pow(2).sum(dim=1, keepdim=True) - 2 * torch.matmul(flatten, self.embed) + self.embed.pow(2).sum(dim=1, keepdim=True) # Origin: (x - embed)^2, shape = [batch_size * length]
        soft_codes = -dist
        _, embed_indices = torch.max(soft_codes, dim=1)
        quantize = self.embed_code(embed_indices.reshape((batch_size, length)))

        if self.balancing_heuristic:
            if self.codes is None:
                self.codes = embed_indices
            else:
                self.codes = torch.cat([self.codes, embed_indices])
                if len(self.codes) > self.max_codes:
                    self.codes = self.codes[-self.max_codes:]
                    self.codes_full = True
        
        if self.training:
            embed_onehot = F.one_hot(embed_indices, self.n_embed) # (batch_size * length, n_embed)
            embed_onehot_sum = embed_onehot.sum(dim=0) # (n_embed)
            embed_sum = torch.matmul(flatten.transpose(0, 1), embed_onehot) # (dim, n_embed)

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1-self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embed * self.eps)) * n
            embed_normalized = self.embed_avg / cluster_size.squeeze()
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        if return_soft_codes:
            return quantize, diff, embed_indices, soft_codes.reshape((batch_size, length, self.dim))
        elif self.new_retun_order:
            return quantize, embed_indices, diff
        else:
            return quantize, diff, embed_indices

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x) + x
        return x

class DiscreteVAE(nn.Module):
    def __init__(self, spec_dim: int, inner_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.codebook = VectorQuantization(dim=inner_dim, n_embed=num_tokens)
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(spec_dim, inner_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(inner_dim, num_tokens, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            ResBlock(num_tokens, kernel_size=3),
            ResBlock(num_tokens, kernel_size=3),
            ResBlock(num_tokens, kernel_size=3),
            nn.Conv1d(num_tokens, inner_dim, kernel_size=1, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass