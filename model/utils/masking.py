import torch

from typing import Optional

def generate_padding_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    '''
        lengths: Tensor, shape = (batch_size)
        Return Padding Mask with shape = (batch_size, max_length)
    '''
    if max_length is None:
        max_length = lengths.max()
    
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device).unsqueeze(0) # shape = [1, max_length]

    return lengths.unsqueeze(dim=-1) > x

def generate_look_ahead_mask(lengths: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    '''
        lengths: Tensor, shape = (batch_size)
        Return Padding Mask with shape = (batch_size, max_length, max_length)
    '''
    if max_length is None:
        max_length = lengths.max()

    lower_trig_mask = torch.tril(torch.ones((max_length, max_length))).to(dtype=torch.bool, device=lengths.device)
    padding_mask = generate_padding_mask(lengths, max_length)

    look_ahead_mask = torch.min(lower_trig_mask, padding_mask.unsqueeze(1))
    
    return look_ahead_mask