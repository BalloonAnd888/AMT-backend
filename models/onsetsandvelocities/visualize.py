import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from preprocessing.constants import *
from preprocessing.mel import MelSpectrogram

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_prediction(model, dataset, device=DEVICE, save_path='prediction_vis.png'):
    idx = np.random.randint(len(dataset))
    data = dataset[idx]
    print(f"Visualizing prediction for: {data['path']}")

    audio = data['audio'].to(device)
    onset = data['onset'].to(device)
    frame = data['frame'].to(device)
    velocity = data['velocity'].to(device)

    mel_extractor = MelSpectrogram().to(device)
    mel = mel_extractor(audio)
    
    if mel.dim() == 3:
        mel = mel.squeeze(0)

    model.eval()
    with torch.no_grad():
        mel_batch = mel.unsqueeze(0)
        pred_onset_stack, pred_vels = model(mel_batch, trainable_onsets=False)
        
        pred_onset = torch.sigmoid(pred_onset_stack[-1]).squeeze(0)
        pred_vel = torch.sigmoid(pred_vels).squeeze(0)

    mel_np = mel.cpu().numpy()
    onset_np = onset.cpu().numpy().T
    frame_np = frame.cpu().numpy().T
    velocity_np = velocity.cpu().numpy().T
    pred_onset_np = pred_onset.cpu().numpy()
    pred_vel_np = pred_vel.cpu().numpy()

    fig, ax = plt.subplots(6, 1, figsize=(16, 24), sharex=True)
    
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
    ax[0].set_title('Input: Mel Spectrogram')

    duration = audio.shape[-1] / SAMPLE_RATE
    extent = [0, duration, 0, 88]
    
    ax[1].imshow(frame_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
    ax[1].set_title('Target: Frames')
    ax[1].set_ylabel('Key')

    ax[2].imshow(onset_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
    ax[2].set_title('Target: Onsets')
    ax[2].set_ylabel('Key')

    ax[3].imshow(velocity_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
    ax[3].set_title('Target: Velocity')
    ax[3].set_ylabel('Key')

    ax[4].imshow(pred_onset_np, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
    ax[4].set_title('Prediction: Onsets')
    ax[4].set_ylabel('Key')

    ax[5].imshow(pred_vel_np, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
    ax[5].set_title('Prediction: Velocity')
    ax[5].set_xlabel('Time (s)')
    ax[5].set_ylabel('Key')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
