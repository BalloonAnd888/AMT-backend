import gradio as gr
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
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
    
    if model_choice == "of":
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
