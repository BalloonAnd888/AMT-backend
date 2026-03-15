import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm

from models.endtoend.endtoend import ETE
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.onsetsandframes.midi import save_midi

def midi_to_note_name(midi_number):
    """Converts a MIDI number to a note name (e.g., 60 -> C4)."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"

def inference(model_path):
    print(f"Device: {DEVICE}")
    
    # 1. Load Model
    model = ETE(
        input_shape=1,
        output_shape=N_KEYS
    ).to(DEVICE)

    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

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
    onset_target = sample['onset'].to(DEVICE).float() # (Time, Keys)

    # 4. Prepare Input (Similar to train.py)
    # Calculate Mel Spectrogram for the whole audio
    full_mels = mel_extractor(audio.unsqueeze(0)).unsqueeze(1) # (1, 1, F, T)

    window_size = full_mels.shape[-1]
    half_window = window_size // 2
    padded_mels = torch.nn.functional.pad(full_mels, (half_window, half_window), mode='constant', value=full_mels.min())

    print(f"Audio Duration: {audio.shape[0]/SAMPLE_RATE:.2f}s")
    print(f"Mel Shape: {full_mels.shape}")
    print(f"Window Size: {window_size}")

    # 5. Sliding Window Inference
    # We want to predict for every time step t in the original mel spectrogram
    num_frames = full_mels.shape[-1]

    # Create batch of windows
    all_windows = []
    for t in range(num_frames):
        # Extract window centered at t
        # padded_mels is (1, 1, F, T + 2*half)
        # slice t : t + window_size
        win = padded_mels[:, :, :, t : t + window_size]
        all_windows.append(win)

    all_windows = torch.cat(all_windows, dim=0) # (T, 1, F, W)

    print(f"Running inference on {num_frames} frames...")

    batch_size = 32
    all_probs = []

    with torch.no_grad():
        for i in tqdm(range(0, num_frames, batch_size), desc="Inference"):
            batch_input = all_windows[i : i + batch_size]
            logits = model(batch_input)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)

    # Concatenate all predictions
    pred_probs = torch.cat(all_probs, dim=0) # (T, Keys)

    # 6. Visualization
    onset_target_np = onset_target.cpu().numpy()
    pred_probs_np = pred_probs.cpu().numpy()
    mel_np = full_mels.squeeze().cpu().numpy()

    print(f"Prediction Shape: {pred_probs_np.shape}")
    print(f"Max Probability: {pred_probs_np.max():.4f}")

    # Save MIDI
    # Simple decoding: threshold > 0.5, fixed duration
    pitches, intervals, velocities = [], [], []
    hop_time = HOP_LENGTH / SAMPLE_RATE
    min_midi = 21
    
    rows, cols = np.nonzero(pred_probs_np > 0.5)
    for r, c in zip(rows, cols):
        # r is time, c is key
        pitches.append(440.0 * (2.0 ** ((c + min_midi - 69.0) / 12.0)))
        start = r * hop_time
        intervals.append([start, start + 0.1]) # Short duration for onsets
        velocities.append(float(pred_probs_np[r, c]))

    midi_path = os.path.join(os.path.dirname(model_path), f'sample_{idx}_pred.mid')
    save_midi(midi_path, np.array(pitches), np.array(intervals), np.array(velocities))
    print(f"Saved MIDI to {midi_path}")

    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. Mel Spectrogram
    librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
    ax[0].set_title('Input: Mel Spectrogram')

    # 2. Target Onsets
    librosa.display.specshow(onset_target_np.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[1], cmap='Greys', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Onsets')
    ax[1].set_ylabel('Key')

    # 3. Predicted Onsets
    librosa.display.specshow(pred_probs_np.T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[2], cmap='Greys', vmin=0, vmax=1)
    ax[2].set_title('Predicted Onsets (Sliding Window)')
    ax[2].set_ylabel('Key')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Attempt to find the latest model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")

    # Find latest model
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'ete_model' in f]
    if model_files:
        model_files.sort()
        latest_model = model_files[-1]
        MODEL_PATH = os.path.join(model_dir, latest_model)
    else:
        MODEL_PATH = os.path.join(model_dir, 'ete_model_20260203_195050.pt')

    inference(MODEL_PATH)
