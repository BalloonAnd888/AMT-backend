import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from mir_eval.util import midi_to_hz

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.decoder import OnsetVelocityNmsDecoder
from models.onsetsandframes.midi import save_midi
from models.ov.inference import strided_inference
from preprocessing.constants import DATA_PATH, N_KEYS, SAMPLE_RATE, HOP_LENGTH, N_MELS, MIN_MIDI, SEQUENCE_LENGTH
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD = (200, 200)
BATCH_NORM = 0
LEAKY_RELU_SLOPE = 0.1
DROPOUT = 0
IN_CHANS = 2

def load_model(model_path):
    """Initializes the OV model and loads weights."""
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
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model.eval()
    return model

def get_decoder():
    """Initializes the onset/velocity decoder."""
    return OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1,
        gauss_conv_ksize=11,
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

def generate_midi(model, decoder, audio_tensor, output_path):
    """Takes an audio tensor, runs forward pass, and writes the resulting predictions to a MIDI file."""
    mel_extractor = MelSpectrogram().to(DEVICE)
    mel = mel_extractor(audio_tensor)
    
    # Wrapper for strided inference to extract and process specific outputs
    def model_wrapper(x):
        with torch.no_grad():
            pred_onset_stack, pred_vels = model(x, trainable_onsets=False)
            # Pad back to T instead of T-1 for strided inference assertion
            probs_pad = F.pad(torch.sigmoid(pred_onset_stack[-1]), (1, 0))
            vels_pad = F.pad(torch.sigmoid(pred_vels), (1, 0))
            return [probs_pad, vels_pad]
            
    probs, vels = strided_inference(model_wrapper, mel, chunk_size=10000, chunk_overlap=100)
    probs = probs.to(DEVICE)
    vels = vels.to(DEVICE)
    print(probs)
    
    # Decode
    df = decoder(probs, vels, pthresh=0.2)
    print(df)
    
    if not df.empty:
        pitches = np.array([midi_to_hz(MIN_MIDI + k) for k in df["key"].values])
        scaling = HOP_LENGTH / SAMPLE_RATE
        onsets = df["t_idx"].values * scaling
        intervals = np.column_stack([onsets, onsets + 0.1]) # Hardcode slight duration as OV doesn't output offsets
        velocities = df["vel"].values # save_midi uses [0, 1] range to scale properly
        
        save_midi(output_path, pitches, intervals, velocities)
        print(f"Successfully saved MIDI to {output_path}\n")
    else:
        print("No notes detected. Could not save MIDI.\n")

def infer_dataset(model_path, data_path=DATA_PATH, num_samples=1):
    """Runs inference on random samples taken from the MAESTRO test set."""
    model = load_model(model_path)
    decoder = get_decoder()
    
    test_dataset = MAESTRO(path=data_path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty.")
        return

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    for _ in range(num_samples):
        idx = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[idx]
        audio = sample['audio'].unsqueeze(0).to(DEVICE)
        
        midi_path = os.path.join(results_dir, f'dataset_sample_{idx}_pred.mid')
        print(f"Running inference on test dataset sample index: {idx}...")
        generate_midi(model, decoder, audio, midi_path)

def infer_audio(model_path, audio_path):
    """Runs inference on a specific audio file."""
    model = load_model(model_path)
    decoder = get_decoder()
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found at {audio_path}")
        return
        
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(DEVICE)
    
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    midi_path = os.path.join(results_dir, f'{filename}_pred.mid')
    
    print(f"Running inference on audio file {audio_path}...")
    generate_midi(model, decoder, audio_tensor, midi_path)

if __name__ == "__main__":
    # Example Usage
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    
    # 1. Run inference on random files from the dataset
    print("--- Testing on Dataset ---")
    infer_dataset(MODEL_PATH, num_samples=1)
    
    # 2. Run inference on a specific file (Uncomment and replace with your actual path)
    # print("--- Testing on Custom Audio File ---")
    # AUDIO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'audio', 'example.wav')
    # infer_audio(MODEL_PATH, AUDIO_FILE_PATH)