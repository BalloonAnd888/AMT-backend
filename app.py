from typing import List

import gradio as gr
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import torch.nn.functional as F
from PIL import Image
import os
import urllib.request
import numpy as np
import tempfile
from mir_eval.util import midi_to_hz
from preprocessing.mel import MelSpectrogram
from preprocessing.constants import SAMPLE_RATE, HOP_LENGTH, N_MELS, N_KEYS, MIN_MIDI
from models.onsetsandframes.of import OnsetsAndFrames
from models.onsetsandframes.decoding import extract_notes
from models.onsetsandframes.midi import save_midi
from models.pianotranscriptionbytedance.inference import PianoTranscription
from torchaudio.transforms import MelSpectrogram as TorchMelSpec, AmplitudeToDB
from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.inference import strided_inference, OnsetVelocityNmsDecoder

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelFiles")

OV_MODEL_PATH = os.path.join(MODEL_DIR, "OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt")
ETE_MODEL_PATH = os.path.join(MODEL_DIR, "ete_model_20260203_195050.pt")
HIGH_RESOLUTION_PATH = os.path.join(MODEL_DIR,
"note_F1=0.9677_pedal_F1=0.9186.pt")
OF_MODEL_PATH = os.path.join(MODEL_DIR,
"onsetsandframes-260209-204911-5000.pt")

MODELS = {
    "ov": {
        "name": "OnsetsAndVelocities",
        "path": OV_MODEL_PATH,
    },
    "of": {
        "name": "OnsetsAndFrames",
        "path": OF_MODEL_PATH,
    },
    "endtoend": {
        "name": "EndToEnd",
        "path": ETE_MODEL_PATH,
    },
    "high_resolution": {
        "name": "HighResolution",
        "path": HIGH_RESOLUTION_PATH,
    },
}

def get_models():
    """Returns the models in a format ideal for a frontend dropdown menu."""
    return [
        {"value": key, "label": model["name"]} 
        for key, model in MODELS.items()
    ]

def process_audio(audio_path, model_choice):
    if audio_path is None:
        return None, None
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_tensor = torch.from_numpy(y).to(device)
    
    mel_spectrogram = MelSpectrogram().to(device)

    log_mel = mel_spectrogram(audio_tensor)
        
    S = log_mel.squeeze().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Log-Mel Spectrogram')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    mel_image = Image.open(buf)
    midi_file_path = None
    model_path = MODELS[model_choice]["path"]
    
    if model_choice == "ov":
        OV_SAMPLE_RATE = 16000
        OV_WINDOW_LENGTH = 2048
        OV_HOP_LENGTH = 384
        OV_N_MELS = 229
        OV_MEL_FMIN = 50
        OV_MEL_FMAX = 8000

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        CONV1X1_HEAD: List[int] = (200, 200)
        BATCH_NORM = 0
        LEAKY_RELU_SLOPE: float = 0.1
        DROPOUT = 0
        IN_CHANS = 2

        INFERENCE_CHUNK_SIZE: float = 400  
        INFERENCE_CHUNK_OVERLAP: float = 11 
        INFERENCE_THRESHOLD: float = 0.74

        SECS_PER_FRAME = OV_HOP_LENGTH / OV_SAMPLE_RATE
        CHUNK_SIZE = round(INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
        CHUNK_OVERLAP = round(INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)

        class TorchWavToLogmel(torch.nn.Module):
            """
            Matches the TorchWavToLogmel used during training in ov_piano/utils.py.
            Uses torchaudio MelSpectrogram + AmplitudeToDB(top_db=80).
            """
            def __init__(self, samplerate, winsize, hopsize, n_mels,
                        mel_fmin=50, mel_fmax=8000):
                super().__init__()
                self.melspec = TorchMelSpec(
                    samplerate, winsize, hop_length=hopsize,
                    f_min=mel_fmin, f_max=mel_fmax, n_mels=n_mels,
                    power=2, window_fn=torch.hann_window)
                self.to_db = AmplitudeToDB(stype="power", top_db=80.0)
                # Must run once to avoid NaNs on first real call
                self.melspec(torch.rand(winsize * 10))

            def forward(self, wav_arr):
                """
                :param wav_arr: 1D float tensor or (chans, time)
                :returns: log-mel spectrogram of shape (n_mels, t)
                """
                mel = self.melspec(wav_arr)
                log_mel = self.to_db(mel)
                return log_mel

        def load_model(model, path, eval_phase=True):
            """Load model weights from checkpoint."""
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=True)
            if eval_phase:
                model.eval()
            else:
                model.train()
            print(f"Loaded {len(state_dict)} parameter tensors from {path}")
            print(f"First param norm: {list(model.parameters())[0].norm().item():.4f}")


        def make_triple_onsets(onsets):
            """
            :param onsets: boolean array of shape (k, t)
            :returns: boolean array of same shape, but every true entry at time
            t is also extended to t+1, t+2.
            """
            result = onsets.copy()
            result[:, 1:] |= result[:, :-1]
            result[:, 1:] |= result[:, :-1]
            return result


        def df_to_midi(df, secs_per_frame, output_path="output.mid"):
            """
            Convert decoder output DataFrame to a MIDI file.

            :param df: DataFrame with columns [batch_idx, key, t_idx, prob, vel]
            :param secs_per_frame: seconds per mel frame (HOP_LENGTH / SAMPLE_RATE)
            :param output_path: path to save MIDI file
            """
            try:
                import pretty_midi
            except ImportError:
                print("pretty_midi not installed. Run: pip install pretty_midi")
                return

            midi = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

            df_sorted = df.sort_values("t_idx")

            for _, row in df_sorted.iterrows():
                onset_time = float(row["t_idx"]) * secs_per_frame
                velocity = int(float(row["vel"]) * 127)
                pitch = int(row["key"]) + MIN_MIDI  # key 0 = A0 = MIDI 21

                note = pretty_midi.Note(
                    velocity=max(1, min(127, velocity)),
                    pitch=max(0, min(127, pitch)),
                    start=onset_time,
                    end=onset_time + 0.2  # default duration; model only predicts onsets
                )
                piano.notes.append(note)

            midi.instruments.append(piano)
            midi.write(output_path)
            print(f"MIDI saved to {output_path} ({len(df_sorted)} notes)")

        model = OnsetsAndVelocities(
            in_chans=IN_CHANS,
            in_height=OV_N_MELS,
            out_height=N_KEYS,
            conv1x1head=CONV1X1_HEAD,
            bn_momentum=BATCH_NORM,
            leaky_relu_slope=LEAKY_RELU_SLOPE,
            dropout_drop_p=DROPOUT
        ).to(DEVICE)

        load_model(model, model_path, eval_phase=True)

        decoder = OnsetVelocityNmsDecoder(
            num_keys=N_KEYS,
            nms_pool_ksize=3,
            gauss_conv_stddev=1,
            gauss_conv_ksize=11,
            vel_pad_left=1,
            vel_pad_right=1
        ).to(DEVICE)

        mel_extractor = TorchWavToLogmel(
            samplerate=OV_SAMPLE_RATE,
            winsize=OV_WINDOW_LENGTH,
            hopsize=OV_HOP_LENGTH,
            n_mels=OV_N_MELS,
            mel_fmin=OV_MEL_FMIN,
            mel_fmax=OV_MEL_FMAX
        ).to(DEVICE)

        print(f"Loading audio from: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=OV_SAMPLE_RATE, mono=True)
        audio = torch.FloatTensor(audio).to(DEVICE)
        audio = audio / (audio.abs().max() + 1e-8)  # peak normalize

        triple_onsets = None # Ground truth not available for arbitrary custom audio files
        with torch.no_grad():
            mel = mel_extractor(audio)       
            mel = mel.unsqueeze(0)              

        print(f"Mel shape: {mel.shape}")
        print(f"Mel range: {mel.min().item():.2f} to {mel.max().item():.2f} dB")

        def model_inference(x):
            """
            x: (1, n_mels, t) chunk from strided_inference
            """
            with torch.no_grad():
                probs, vels = model(x, trainable_onsets=False)
                probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
                vels = F.pad(torch.sigmoid(vels), (1, 0))
            return probs, vels

        with torch.no_grad():
            onset_pred, vel_pred = strided_inference(
                model_inference, mel, CHUNK_SIZE, CHUNK_OVERLAP)

            onset_pred = onset_pred.to(DEVICE)
            vel_pred = vel_pred.to(DEVICE)

            print(f"Max onset prob: {onset_pred.max().item():.4f}")
            print(f"Mean onset prob: {onset_pred.mean().item():.4f}")

            # Decode onsets
            df = decoder(onset_pred, vel_pred, INFERENCE_THRESHOLD)
            print(f"Decoded {len(df)} onsets")
            if len(df) > 0:
                print(df.head(10))

            onset_pred_np = onset_pred.cpu().numpy().squeeze()  # (88, t)
            vel_pred_np = vel_pred.cpu().numpy().squeeze()       # (88, t)

        # Create a temporary MIDI file
        temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi_file_path = temp_midi.name
        temp_midi.close()

        if len(df) > 0:
            df_to_midi(df, SECS_PER_FRAME, midi_file_path)
            print(f"Saved results to {midi_file_path}")

    elif model_choice == "of":
        model = OnsetsAndFrames(
            input_features=N_MELS, 
            output_features=N_KEYS, 
            model_complexity=48).to(device)
            
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Model file not found at {model_path}")
            
        model.eval()
        with torch.no_grad():
            onset_pred, offset_pred, _, frame_pred, velocity_pred = model(log_mel)
            
        p_est, i_est, v_est = extract_notes(
            onset_pred.squeeze(0), 
            frame_pred.squeeze(0), 
            velocity_pred.squeeze(0)
        )
        
        scaling = HOP_LENGTH / SAMPLE_RATE
        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
        
        temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
        midi_file_path = temp_midi.name
        temp_midi.close()
        
        save_midi(midi_file_path, p_est, i_est, v_est)
        print(f"Saved results to {midi_file_path}")
        
    elif model_choice == "high_resolution":
        try:
            if not os.path.exists(model_path):
                print(f"Downloading checkpoint to {model_path}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                urllib.request.urlretrieve('https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth', model_path)

            temp_midi = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
            midi_file_path = temp_midi.name
            temp_midi.close()
            
            hr_audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            transcriptor = PianoTranscription(device=device, checkpoint_path=model_path)
            transcriptor.transcribe(hr_audio, midi_file_path)
        except Exception as e:
            print(f"ByteDance High Resolution module failed: {e}")

    return mel_image, midi_file_path

# Build dropdown choices as a list of (label, value) tuples for Gradio
dropdown_choices = [(model["name"], key) for key, model in MODELS.items()]

with gr.Blocks(title="AMT Backend - Music Transcription") as demo:
    gr.Markdown("# AMT Backend - Music Transcription")
    gr.Markdown("Upload an audio file and select a model to generate the transcription.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            model_input = gr.Dropdown(choices=dropdown_choices, value="ov", label="Select Model")
            transcribe_btn = gr.Button("Transcribe")
            
        with gr.Column():
            mel_output = gr.Image(type="filepath", label="Mel Spectrogram")
            midi_output = gr.File(label="Output MIDI")
            
    transcribe_btn.click(
        fn=process_audio,
        inputs=[audio_input, model_input],
        outputs=[mel_output, midi_output],
        api_name="transcribe"
    )
    
    # Hidden component to expose the models endpoint via Gradio API
    models_output = gr.JSON(visible=False)
    demo.load(
        fn=get_models,
        inputs=[],
        outputs=[models_output],
        api_name="models"
    )

if __name__ == "__main__":
    demo.launch()
