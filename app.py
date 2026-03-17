import gradio as gr
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
from preprocessing.mel import MelSpectrogram
from preprocessing.constants import SAMPLE_RATE, HOP_LENGTH

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
        
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_tensor = torch.from_numpy(y)
    
    mel_spectrogram = MelSpectrogram()

    log_mel = mel_spectrogram(audio_tensor)
        
    S = log_mel.squeeze().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Log-Mel Spectrogram')
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    # Return the spectrogram image and None for the MIDI file output
    return Image.open(buf), None

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
