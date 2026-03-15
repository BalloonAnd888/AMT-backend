import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from mir_eval.util import midi_to_hz
from tqdm import tqdm

from models.onsetsandframes.of import OnsetsAndFrames
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, MIN_MIDI, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX, N_MELS
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.onsetsandframes.decoding import extract_notes
from models.onsetsandframes.midi import save_midi
from models.onsetsandframes.utils import save_pianoroll

def inference(model_path, audio_path=None):
    print(f"Device: {DEVICE}")
    
    model = OnsetsAndFrames(
        input_features=N_MELS, 
        output_features=N_KEYS, 
        model_complexity=48).to(DEVICE)
    
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    model.eval()
    mel_extractor = MelSpectrogram().to(DEVICE)

    if audio_path:
        print(f"Inference on audio file: {audio_path}")
        idx = os.path.splitext(os.path.basename(audio_path))[0]
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio = torch.tensor(audio).to(DEVICE)
        batch = {
            'audio': audio.unsqueeze(0)
        }
        onset_target = None
        frame_target = None
        velocity_target = None
    else:
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

        batch = {
            'audio': sample['audio'].unsqueeze(0).to(DEVICE),
            'onset': sample['onset'].unsqueeze(0).to(DEVICE).float(),
            'offset': sample['offset'].unsqueeze(0).to(DEVICE).float(),
            'frame': sample['frame'].unsqueeze(0).to(DEVICE).float(),
            'velocity': sample['velocity'].unsqueeze(0).to(DEVICE).float()
        }

        onset_target = batch['onset'].squeeze(0)
        frame_target = batch['frame'].squeeze(0)
        velocity_target = batch['velocity'].squeeze(0)

    audio = batch['audio'].squeeze(0)

    mel = mel_extractor(audio)
    print(mel.shape)

    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
        'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
        'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    }

    # print(predictions['onset'].shape)
    # print(predictions['onset'])

    p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'])

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    if onset_target is not None:
        save_pianoroll(os.path.join(results_dir, f'sample_{idx}_gt.png'), onset_target, frame_target)
    save_pianoroll(os.path.join(results_dir, f'sample_{idx}_pred.png'), predictions['onset'], predictions['frame'])
    save_midi(os.path.join(results_dir, f'sample_{idx}_pred.mid'), p_est, i_est, v_est)
    print(f"Saved results to {results_dir}")

    # Visualization
    mel_np = mel.squeeze(0).cpu().numpy()
    onset_pred_np = predictions['onset'].cpu().detach().numpy().T
    frame_pred_np = predictions['frame'].cpu().detach().numpy().T

    if onset_target is not None:
        onset_target_np = onset_target.cpu().numpy().T
        frame_target_np = frame_target.cpu().numpy().T
        
        fig, ax = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

        librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
        ax[0].set_title('Input: Mel Spectrogram')

        librosa.display.specshow(onset_target_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[1], cmap='Greys', vmin=0, vmax=1)
        ax[1].set_title('Ground Truth Onsets')

        librosa.display.specshow(frame_target_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[2], cmap='Greys', vmin=0, vmax=1)
        ax[2].set_title('Ground Truth Frames')

        librosa.display.specshow(onset_pred_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[3], cmap='Greys', vmin=0, vmax=1)
        ax[3].set_title('Predicted Onsets')

        librosa.display.specshow(frame_pred_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[4], cmap='Greys', vmin=0, vmax=1)
        ax[4].set_title('Predicted Frames')
    else:
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        librosa.display.specshow(mel_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=MEL_FMIN, fmax=MEL_FMAX, ax=ax[0], cmap='magma')
        ax[0].set_title('Input: Mel Spectrogram')

        librosa.display.specshow(onset_pred_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[1], cmap='Greys', vmin=0, vmax=1)
        ax[1].set_title('Predicted Onsets')

        librosa.display.specshow(frame_pred_np, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='cqt_note', fmin=librosa.note_to_hz('A0'), ax=ax[2], cmap='Greys', vmin=0, vmax=1)
        ax[2].set_title('Predicted Frames')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    # Attempt to find the latest model in the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models")
    audio_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'audio'))

    MODEL_PATH = os.path.join(model_dir, 'onsetsandframes-260209-204911-5000.pt')
    
    audio_path = os.path.join(audio_dir, 'route1.wav')
    
    inference(MODEL_PATH, audio_path)
