import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional, Callable, Union

class MelSpectrogram(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 n_mels: int = 128,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 pad: int = 0,
                 center: bool = True,
                 power: Optional[float] = 2.0,
                 normalized: Union[bool, str] = False,
                 pad_mode: str = 'reflect',
                 onesided: bool = True,
                 norm: Optional[str] = None,
                 mel_scale: str = 'htk') -> None:
        super().__init__()
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2

        self.f_max = f_max if f_max is not None else float(sample_rate // 2)

        self.spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=window_fn,
            pad=pad,
            center=center,
            power=power,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided
        )

        self.mel_scale = MelScale(
            n_stft= n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            norm=norm,
            mel_scale=mel_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(x)
        x = self.mel_scale(x)
        return x

class Spectrogram(nn.Module):
    def __init__(self,
                 n_fft: int,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., torch.Tensor] = torch.hann_window,
                 pad: int = 0,
                 center: bool = True,
                 power: Optional[float] = 2.0,
                 normalized: Union[bool, str] = False,
                 pad_mode: str = 'reflect',
                 onesided: bool = True) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2

        self.pad = pad
        self.center = center
        self.power = power
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.register_buffer("window", window_fn(self.win_length))

        self.num_pad = int((self.n_fft - self.hop_length) / 2)
        self.frame_length_norm = False
        self.window_norm = False

        self.get_spec_norm()
    
    def get_spec_norm(self):
        if torch.jit.isinstance(self.normalized, str):
            if self.normalized == 'frame_length':
                self.frame_length_norm = True
            elif self.normalized == 'window':
                self.window_norm = True
        elif torch.jit.isinstance(self.normalized, bool):
            if self.normalized:
                self.window_norm = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: Waveform input, shape = [batch_size, time]
            ----
            output: Spectrogram shape = [batch_size, n_fft // 2 + 1, time]
        '''
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad), mode='constant')
        
        if self.center == False:
            x = F.pad(x, (self.num_pad, self.num_pad), mode='reflect')

        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.frame_length_norm,
            onesided=self.onesided,
            return_complex=True
        )

        if self.window_norm:
            x = x / self.window.power(2.0).sum().sqrt()
        
        if self.power is not None:
            x = x.abs().pow(self.power)

        return x

class MelScale(nn.Module):
    def __init__(self,
                 n_stft: int,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk") -> None:
        super().__init__()
        assert mel_scale in ['htk', 'slaney'], "Invalid Format of Scaling"

        self.sample_rate = sample_rate
        self.n_stft = n_stft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.norm = norm
        self.mel_scale = mel_scale
        self.f_sp = 200.0 / 3
        self.logstep = math.log(6.4) / 27.0

        self.register_buffer('filterbank', self.mel_filterbank())

    def hz_to_mel(self, freq: float) -> float:
        if self.mel_scale == 'htk':
            return 2595.0 * math.log10(1.0 + (freq / 700))
        else:
            if freq < 1000:
                return freq / self.f_sp
            else:
                return 15.0 + (math.log(freq / 1000) / self.logstep)
    
    def mel_to_hz(self, mels: torch.Tensor) -> torch.Tensor:
        if self.mel_scale == 'htk':
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
        else:
            freqs = mels * self.f_sp
            is_over = (mels >= 15)
            freqs[is_over] = 1000.0 * torch.exp((mels[is_over] - 15.0) * self.logstep)
            return freqs
    
    def create_triangular_filterbank(self, all_freqs: torch.Tensor, f_pts: torch.Tensor):
        f_diff = f_pts[1:] - f_pts[:-1] # (n_mels - 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1) # (n_stft, n_mels)

        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
        up_slopes = slopes[:, 2:] / f_diff[1:]
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        return fb
    
    def mel_filterbank(self):
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        mel_min = self.hz_to_mel(self.f_min)
        mel_max = self.hz_to_mel(self.f_max)

        mel_pts = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        f_pts = self.mel_to_hz(mel_pts)

        filterbank = self.create_triangular_filterbank(all_freqs, f_pts)

        if self.norm is not None and self.norm == 'slaney':
            enorm = 2.0 / (f_pts[2 : self.n_mels + 2] - f_pts[: self.n_mels])
            filterbank = filterbank * enorm.unsqueeze(0)

        return filterbank # (n_stft, n_mels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: Spectrogram input, shape = [batch_size, n_stft, time]
            -----
            output: Mel - Spectrogram, shape = [batch_size, n_mels, time]
        '''
        x = torch.matmul(x.transpose(1, 2), self.filterbank).transpose(1, 2)
        return x