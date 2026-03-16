import gradio as gr
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
from preprocessing.mel import MelSpectrogram
from preprocessing.constants import SAMPLE_RATE, HOP_LENGTH

def process_audio(audio_path):
    if audio_path is None:
        return None
        
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
    
    return Image.open(buf)

demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    outputs=gr.Image(type="filepath", label="Mel Spectrogram"),
    title="AMT Backend - Mel Spectrogram Generator",
    description="Upload an audio file to generate its Log-Mel Spectrogram."
)

if __name__ == "__main__":
    demo.launch()
