import torch
import torch.nn.functional as F
import torchaudio

import numpy as np

import re
import librosa
from scipy.io import wavfile
import json

from typing import Optional, Union, List

MAX_AUDIO_VALUE = 32768

class XTTSProcessor:
    def __init__(self,
                 tokenizer_path: str, pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>",
                 # Reference Audio config
                 speaker_sampling_rate: int = 16000, speaker_n_mels: int = 64, speaker_n_fft: int = 512, speaker_hop_length: int = 160, speaker_win_length: int = 400, pre_emphasis: float = 0.97,
                 # Grouth-True Audio Config,
                 gt_sampling_rate: int = 22050, gt_n_mels: int = 80, gt_n_fft: int = 1024, gt_hop_length: int = 256, gt_win_length: int = 1024,
                 # Device
                 device: Union[str, int] = 'cpu') -> None:
        # Text
        patterns = json.load(open(tokenizer_path, 'r', encoding='utf8'))
        self.yolo = patterns
        self.replace_dict = patterns['replace']
        self.mapping = patterns['mapping']
        vocab = []

        for key in patterns.keys():
            if key == 'replace' or key == 'mapping':
                continue
            vocab += patterns[key]
            
        self.dictionary = self.create_vocab_dictionary(vocab, pad_token, delim_token, unk_token)

        self.pattern = self.sort_pattern(vocab + list(patterns['mapping'].keys()))

        self.pad_token = pad_token
        self.delim_token = delim_token
        self.unk_token = unk_token

        self.pad_id = self.find_token_id(pad_token)
        self.delim_id = self.find_token_id(delim_token)
        self.unk_id = self.find_token_id(unk_token)

        # Reference Audio
        self.filter = torch.FloatTensor([-pre_emphasis, 1.0]).unsqueeze(0).unsqueeze(0).to(device)
        self.speaker_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=speaker_sampling_rate,
            n_fft=speaker_n_fft,
            hop_length=speaker_hop_length,
            win_length=speaker_win_length,
            n_mels=speaker_n_mels
        ).to(device)

        # Grouth-true Audio
        self.gt_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=gt_sampling_rate,
            n_fft=gt_n_fft,
            hop_length=gt_hop_length,
            win_length=gt_win_length,
            n_mels=gt_n_mels,
            f_min=0,
            f_max=8000,
            norm="slaney"
        ).to(device)

        self.device = device

    def create_vocab_dictionary(self, vocab: List[str], pad_token: str, delim_token: str, unk_token: str):
        dictionary = []
        dictionary.append(pad_token)

        for item in vocab:
            if item not in dictionary:
                dictionary.append(item)

        dictionary.append(delim_token)
        dictionary.append(unk_token)

        return dictionary
    
    def sort_pattern(self, patterns: List[str]):
        patterns = sorted(patterns, key=len)
        patterns.reverse()

        return patterns
    
    def find_token_id(self, token: str):
        if token in self.dictionary:
            return self.dictionary.index(token)
        return self.dictionary.index(self.unk_token)
    
    def token2text(self, tokens: np.ndarray, get_string: bool = False) -> str:
        words = []
        for token in tokens:
            words.append(self.dictionary[token])

        if get_string:
            return "".join(words).replace(self.delim_token, " ")
        
        return words
    
    def spec_replace(self, word: str):
        for key in self.replace_dict:
            word = word.replace(key, self.replace_dict[key])
        return word
    
    def sentence2tokens(self, sentence: str):
        phonemes = self.sentence2phonemes(sentence)
        tokens = self.phonemes2tokens(phonemes)
        return tokens

    def phonemes2tokens(self, phonemes: List[str]):
        tokens = []
        for phoneme in phonemes:
            tokens.append(self.find_token_id(phoneme))
        return torch.tensor(tokens)
    
    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        # sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 4, reverse: bool = True):
        if len(text) == 1:
            if text in patterns:
                if text in self.mapping:
                    return [self.mapping[text]]
                else:
                    return [text]
            return [self.unk_token]
        if reverse:
            text = [text[i] for i in range(len(text) - 1, -1, -1)]
            text = "".join(text)
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if reverse:
                item = [item[i] for i in range(len(item) - 1, -1, -1)]
                item = "".join(item)
                
            if item in patterns:
                if item in self.mapping:
                    graphemes.append(self.mapping[item])
                else:
                    graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_token)
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        if reverse:
            graphemes = [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]

        return graphemes

    def sentence2phonemes(self, sentence: str):
        sentence = self.spec_replace(self.clean_text(sentence.upper()))
        sentence = self.clean_text(sentence)
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            graphemes += self.slide_graphemes(word, self.pattern, n_grams=4)
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes

    def load_audio(self, path: str, target_sr: Optional[int] = None):
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if target_sr is not None and target_sr != sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        signal = torch.tensor(signal, dtype=torch.float32).to(self.device)
        return signal
    
    def log_mel(self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
    
    def speaker_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = self.speaker_spectrogram(x)
        x = self.log_mel(x)
        return x
    
    def gt_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gt_spectrogram(x)
        x = self.log_mel(x)
        return x
    
    def __call__(self, texts: List[str], signals: List[torch.Tensor]) -> torch.Any:
        pass