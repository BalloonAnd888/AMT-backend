import torch
import os
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

from preprocessing.constants import *
from preprocessing.mel import MelSpectrogram

def visualizeFile(data, save_path='visualize_check.png'):
    print(f"Visualizing: {data['path']}")
    
    audio = data['audio']
    frames = (data['label'] > 1).float()
    onsets = (data['label'] == 3).float()
    
    print(f"Audio shape: {audio.shape}")
    print(f"Label shape: {frames.shape}")

    mel_extractor = MelSpectrogram().to(DEVICE)
    audio_device = audio.to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        mel = mel_extractor(audio_device) 
    
    mel = mel.squeeze(0).cpu().numpy()
    frames_np = frames.cpu().numpy().T 
    onsets_np = onsets.cpu().numpy().T 

    fig, ax = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    filepath = data['path']
    if 'maestro-v3.0.0' in filepath:
        display_name = filepath.split('maestro-v3.0.0')[-1].lstrip(os.sep + '/')
    else:
        display_name = os.path.basename(filepath)
    fig.suptitle(f"File: {display_name}", fontsize=16)

    librosa.display.specshow(
        mel,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        ax=ax[0],
        cmap='magma'
    )
    ax[0].set_title(f"Input: Mel Spectrogram")

    duration = len(audio) / SAMPLE_RATE

    ax[1].imshow(frames_np, origin='lower', aspect='auto', interpolation='nearest', 
                 cmap='Greys', extent=[0, duration, 0, 88])
    ax[1].set_title('Target: Note Frames (Sustain)')
    ax[1].set_ylabel('Piano Key Index (0-87)')
    
    ax[2].imshow(onsets_np, origin='lower', aspect='auto', interpolation='nearest', 
                 cmap='Greys', extent=[0, duration, 0, 88])
    ax[2].set_title('Target: Note Onsets (Attack)')
    ax[2].set_xlabel('Time (seconds)')
    ax[2].set_ylabel('Piano Key Index (0-87)')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")
    
    audio_np = audio.cpu().numpy() 
    audio_save_path = 'visualize_sample.wav'
    sf.write(audio_save_path, audio_np, SAMPLE_RATE)
    print(f"Audio saved to '{audio_save_path}'")
