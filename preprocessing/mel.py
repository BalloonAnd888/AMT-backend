import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from preprocessing.constants import *

class MelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, window_length=WINDOW_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX):
        super().__init__()

        self.window_length = window_length
        self.hop_length = hop_length

        mel_basis = librosa_mel_fn(
            sr=sample_rate,
            n_fft=window_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer('mel_basis', mel_basis)
    
    def forward(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        padding = (self.window_length - self.hop_length) // 2
        audio = F.pad(audio, (padding, padding), mode='reflect')

        stft = torch.stft(
            audio,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            window=torch.hann_window(self.window_length).to(audio.device),
            center=False,
            return_complex=True
        )

        magnitude = torch.abs(stft)

        mel_output = torch.matmul(self.mel_basis, magnitude)

        log_mel = torch.log(torch.clamp(mel_output, min=1e-5))

        return log_mel
    