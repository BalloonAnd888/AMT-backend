import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from torchinfo import summary

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX, N_MELS
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD = (200, 200)
BATCH_NORM = 0.95
LEAKY_RELU_SLOPE = 0.1
DROPOUT = 0.15
IN_CHANS = 2

def inference(model_path):
    print(f"Device: {DEVICE}")

    # 1. Load Model
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    summary(model)
    model.eval()
    mel_extractor = MelSpectrogram().to(DEVICE)

    # 2. Load Dataset
    # Using 'test' group
    test_dataset = MAESTRO(path=DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    
    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty.")
        return

    # 3. Get Random Sample
    idx = random.randint(0, len(test_dataset) - 1)
    print(f"Visualizing sample index: {idx}")

    sample = test_dataset[idx]
    audio = sample['audio'].to(DEVICE) # (Samples)
    # Targets are (Time, Keys)
    onset_target = sample['onset'].to(DEVICE).float() 
    velocity_target = sample['velocity'].to(DEVICE).float()

    # 4. Prepare Input
    # Mel extractor expects (Batch, Samples)
    mel = mel_extractor(audio.unsqueeze(0)) # (1, F, T)
    
    print(f"Audio Duration: {audio.shape[0]/SAMPLE_RATE:.2f}s")
    print(f"Mel Shape: {mel.shape}")

    with torch.no_grad():
        # Forward Pass
        # Returns: x_stages, velocities
        # Model expects (B, F, T)
        pred_onset_stack, pred_vels = model(mel, trainable_onsets=False)
        
        # Get last stage predictions
        pred_onset_logits = pred_onset_stack[-1] # (B, Keys, T-1)
        pred_vel_logits = pred_vels # (B, Keys, T-1)

        # Apply sigmoid
        pred_onset_probs = torch.sigmoid(pred_onset_logits)
        pred_vel_probs = torch.sigmoid(pred_vel_logits)

    # 5. Visualization Preparation
    mel_np = mel.squeeze(0).cpu().numpy()
    
    # Targets need to be transposed to (Keys, Time) for specshow
    onset_target_np = onset_target.cpu().numpy().T
    velocity_target_np = velocity_target.cpu().numpy().T
    
    # Predictions are (1, Keys, T-1). Squeeze batch.
    pred_onset_np = pred_onset_probs.squeeze(0).cpu().numpy()
    pred_vel_np = pred_vel_probs.squeeze(0).cpu().numpy()

    # Align dimensions: Model output is T-1.
    # We slice the targets to match the model output (dropping the first frame, similar to train.py)
    
    # Ensure we don't go out of bounds if shapes differ slightly
    min_len = min(onset_target_np.shape[1] - 1, pred_onset_np.shape[1])
    
    # Slice targets starting from 1 to match T-1
    onset_target_vis = onset_target_np[:, 1 : 1 + min_len]
    velocity_target_vis = velocity_target_np[:, 1 : 1 + min_len]
    
    # Slice predictions
    pred_onset_vis = pred_onset_np[:, :min_len]
    pred_vel_vis = pred_vel_np[:, :min_len]
    
    # Slice Mel (Mel is T)
    mel_vis = mel_np[:, 1 : 1 + min_len]

    print(f"Visualization Length: {min_len} frames")

    # 6. Plot
    fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

    # Mel
    librosa.display.specshow(mel_vis, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
    ax[0].set_title('Input: Mel Spectrogram')

    # Ground Truth Onset
    librosa.display.specshow(onset_target_vis, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[1], cmap='Greys', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Onsets')
    ax[1].set_ylabel('Key')

    # Predicted Onset
    librosa.display.specshow(pred_onset_vis, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[2], cmap='Greys', vmin=0, vmax=1)
    ax[2].set_title('Predicted Onsets (Probabilities)')
    ax[2].set_ylabel('Key')

    # Ground Truth Velocity
    librosa.display.specshow(velocity_target_vis / 127.0, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[3], cmap='magma', vmin=0, vmax=1)
    ax[3].set_title('Ground Truth Velocity')
    ax[3].set_ylabel('Key')

    # Predicted Velocity
    librosa.display.specshow(pred_vel_vis, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[4], cmap='magma', vmin=0, vmax=1)
    ax[4].set_title('Predicted Velocity')
    ax[4].set_ylabel('Key')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Attempt to find the latest model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")

    # audio_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'audio'))

    MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    
    # Find latest model
    # MODEL_PATH = ""
    # if os.path.exists(model_dir):
    #     model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'ov_model' in f]
    #     if model_files:
    #         model_files.sort()
    #         latest_model = model_files[-1]
    #         MODEL_PATH = os.path.join(model_dir, latest_model)
    
    # if not MODEL_PATH:
    #     # Fallback or specific path
    #     MODEL_PATH = os.path.join(model_dir, 'ov_model_latest.pt')

    print(MODEL_PATH)
    inference(MODEL_PATH)