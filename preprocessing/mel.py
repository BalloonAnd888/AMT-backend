import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from preprocessing.constants import *

class MelSpectrogram(torch.nn.Module):
    def __init__(self):
        super().__init__()

        mel_basis = librosa_mel_fn(
            sr=SAMPLE_RATE,
            n_fft=WINDOW_LENGTH,
            n_mels=N_MELS,
            fmin=MEL_FMIN,
            fmax=MEL_FMAX
        )

        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer('mel_basis', mel_basis)
    
    def forward(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        padding = (WINDOW_LENGTH - HOP_LENGTH) // 2
        audio = F.pad(audio, (padding, padding), mode='reflect')

        stft = torch.stft(
            audio,
            n_fft=WINDOW_LENGTH,
            hop_length=HOP_LENGTH,
            window=torch.hann_window(WINDOW_LENGTH).to(audio.device),
            center=False,
            return_complex=True
        )

        magnitude = torch.abs(stft)

        mel_output = torch.matmul(self.mel_basis, magnitude)

        log_mel = torch.log(torch.clamp(mel_output, min=1e-5))

        return log_mel
    