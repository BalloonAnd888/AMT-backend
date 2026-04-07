import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from typing import List
from torchaudio.transforms import MelSpectrogram as TorchMelSpec, AmplitudeToDB

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.inference import strided_inference, OnsetVelocityNmsDecoder
from preprocessing.constants import N_KEYS

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

# Plot constants
FIGSIZE: List[float] = (20, 20)
MEL_CMAP: str = "bone_r"
ROLL_CMAP: str = "bone_r"
TN_RGB: List[int] = (255, 255, 255)
TP_RGB: List[int] = (0, 0, 0)
FP_RGB: List[int] = (179, 179, 255)
FN_RGB: List[int] = (255, 51, 153)
TITLE_SIZE: int = 20
LABEL_SIZE: int = 16
TICK_SIZE: int = 25

# MIDI constants
MIN_MIDI = 21  # MIDI note number of piano key 0 (A0)

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

def qualitative_plot(gt_mel, gt_roll, pred_ons, pred_vel,
                     mel_cmap="bone_r", roll_cmap="bone_r",
                     figsize=(20, 20), threshold=0.74,
                     tn_rgb=(255, 255, 255), tp_rgb=(0, 0, 0),
                     fp_rgb=(179, 179, 255), fn_rgb=(255, 51, 153),
                     secs_per_frame=SECS_PER_FRAME,
                     title_size=20, label_size=16, tick_size=25,
                     ev_title="Ground Truth vs. Thresholded Onset Predictions",
                     min_idx=None, max_idx=None, invert_yaxis=True):
    """
    :param gt_mel: Log-mel spectrogram of shape (f, t)
    :param gt_roll: Ground truth boolean piano roll of shape (k, t), or None
    :param pred_ons: Predicted onsets of shape (k, t) in [0, 1]
    :param pred_vel: Predicted velocities of shape (k, t) in [0, 1]
    :returns: figure and axes
    """
    if max_idx is not None:
        gt_mel = gt_mel[:, :max_idx]
        pred_ons = pred_ons[:, :max_idx]
        pred_vel = pred_vel[:, :max_idx]
        if gt_roll is not None:
            gt_roll = gt_roll[:, :max_idx]
    if min_idx is not None:
        gt_mel = gt_mel[:, min_idx:]
        pred_ons = pred_ons[:, min_idx:]
        pred_vel = pred_vel[:, min_idx:]
        if gt_roll is not None:
            gt_roll = gt_roll[:, min_idx:]

    nrows = 4 if gt_roll is not None else 3
    fig, axes = plt.subplots(nrows=nrows, figsize=figsize, sharex=True)

    mel_ax = axes[0]
    v_ax = axes[1]
    o_ax = axes[2]

    mel_ax.imshow(gt_mel, cmap=mel_cmap, aspect="auto")
    v_ax.imshow(pred_vel, cmap=roll_cmap, aspect="auto")
    o_ax.imshow(pred_ons, cmap=roll_cmap, aspect="auto")

    if gt_roll is not None:
        eval_ax = axes[3]
        pred_mask = (pred_ons >= threshold)
        tp_mask = (gt_roll & pred_mask)
        fp_mask = (~gt_roll & pred_mask)
        fn_mask = (gt_roll & ~pred_mask)

        eval_arr = np.zeros(gt_roll.shape + (3,), dtype=np.uint8)
        eval_arr[:] = tn_rgb
        eval_arr[tp_mask.nonzero()] = tp_rgb
        eval_arr[fp_mask.nonzero()] = fp_rgb
        eval_arr[fn_mask.nonzero()] = fn_rgb
        eval_ax.imshow(eval_arr, aspect="auto")
        eval_ax.set_ylabel("Key", fontsize=label_size)
        eval_ax.set_title(ev_title, fontsize=title_size, pad=20)
        eval_ax.tick_params(labelsize=tick_size)
        eval_ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x * secs_per_frame:g}"))
        eval_ax.set_xlabel("Time (s)", fontsize=label_size)
        if invert_yaxis:
            eval_ax.invert_yaxis()
    else:
        o_ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x * secs_per_frame:g}"))
        o_ax.set_xlabel("Time (s)", fontsize=label_size)

    mel_ax.set_ylabel("Frequency", fontsize=label_size)
    v_ax.set_ylabel("Key", fontsize=label_size)
    o_ax.set_ylabel("Key", fontsize=label_size)

    mel_ax.tick_params(labelsize=tick_size)
    v_ax.tick_params(labelsize=tick_size)
    o_ax.tick_params(labelsize=tick_size)

    mel_ax.set_title("Input Spectrogram", fontsize=title_size, pad=20)
    v_ax.set_title("Predicted Onset Velocities", fontsize=title_size, pad=20)
    o_ax.set_title("Predicted Onset Probabilities", fontsize=title_size, pad=20)

    if invert_yaxis:
        mel_ax.invert_yaxis()
        v_ax.invert_yaxis()
        o_ax.invert_yaxis()

    return fig, axes

def inference(model_path, audio_file_path: str, save_midi=True, show_plot=True):
    """
    Run onset+velocity inference on a specified audio file.

    :param model_path: Path to the .pt model checkpoint.
    :param audio_file_path: Path to the audio file to transcribe.
    :param save_midi: If True, saves decoded onsets as a MIDI file.
    :param show_plot: If True, displays the qualitative plot.
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

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

    print(f"Loading audio from: {audio_file_path}")
    audio, _ = librosa.load(audio_file_path, sr=OV_SAMPLE_RATE, mono=True)
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

    if save_midi and len(df) > 0:
        midi_out = os.path.join(
            results_dir,
            os.path.splitext(os.path.basename(audio_file_path))[0] + '_pred.mid')
        df_to_midi(df, SECS_PER_FRAME, midi_out)

    if show_plot:
        mel_for_plot = mel.cpu().numpy().squeeze()  # (229, t)

        # Align ground truth length to prediction length
        t = onset_pred_np.shape[-1]
        if triple_onsets is not None:
            gt_plot = triple_onsets[:, :t].astype(bool)
        else:
            gt_plot = None

        fig, _ = qualitative_plot(
            mel_for_plot, gt_plot,
            onset_pred_np, vel_pred_np,
            mel_cmap=MEL_CMAP,
            roll_cmap=ROLL_CMAP,
            figsize=FIGSIZE,
            threshold=INFERENCE_THRESHOLD,
            tn_rgb=TN_RGB, tp_rgb=TP_RGB,
            fp_rgb=FP_RGB, fn_rgb=FN_RGB,
            secs_per_frame=SECS_PER_FRAME,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            ev_title="Ground Truth vs. Predictions"
        )
        plt.tight_layout()
        plot_out = os.path.join(results_dir, "qualitative_plot.png")
        plt.savefig(plot_out, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {plot_out}")
        plt.show()

    return df

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(
        model_dir,
        'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')

    print("--- OnsetsAndVelocities Inference with Custom Audio File ---")

    # Specify the audio file path directly
    audio_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'audio'))
    # Replace 'your_audio_file.wav' with the actual name of your audio file in the 'audio' folder
    AUDIO_FILE_PATH = os.path.join(audio_folder_path, 'route1.wav')

    # Run inference with the specified audio file
    df = inference(MODEL_PATH, AUDIO_FILE_PATH, save_midi=True, show_plot=False)
