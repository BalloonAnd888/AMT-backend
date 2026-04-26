import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from typing import List
from collections import defaultdict
from tqdm import tqdm

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.inference import strided_inference, OnsetVelocityNmsDecoder
from preprocessing.constants import N_KEYS, DATA_PATH, SAMPLE_RATE, WINDOW_LENGTH, HOP_LENGTH, N_MELS, MEL_FMAX, MEL_FMIN
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.evaluate import threshold_eval_single_file

OV_SAMPLE_RATE = SAMPLE_RATE
OV_WINDOW_LENGTH = WINDOW_LENGTH
OV_HOP_LENGTH = HOP_LENGTH
OV_N_MELS = N_MELS
OV_MEL_FMIN = MEL_FMIN
OV_MEL_FMAX = MEL_FMAX

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

def evaluate(model_path, data_path=DATA_PATH, save_path=None):
    print(f"Evaluating model on {DEVICE}...")

    csv_path = os.path.join(data_path, "maestro-v3.0.0.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: MAESTRO CSV not found at {csv_path}")
        return
        
    df_maestro = pd.read_csv(csv_path)
    test_files = df_maestro[df_maestro['split'] == 'test']

    if len(test_files) == 0:
        print("ERROR: Test dataset is empty.")
        return

    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=OV_N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    if os.path.exists(model_path):
        load_model(model, model_path, eval_phase=True)
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1,
        gauss_conv_ksize=11,
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

    mel_extractor = MelSpectrogram(
        sample_rate=OV_SAMPLE_RATE,
        window_length=OV_WINDOW_LENGTH,
        hop_length=OV_HOP_LENGTH,
        n_mels=OV_N_MELS,
        fmin=OV_MEL_FMIN,
        fmax=OV_MEL_FMAX
    ).to(DEVICE)

    metrics = defaultdict(list)

    def model_inference(x):
        with torch.no_grad():
            probs, vels = model(x, trainable_onsets=False)
            probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
            vels = F.pad(torch.sigmoid(vels), (1, 0))
        return probs, vels

    for _, row in tqdm(test_files.iterrows(), total=len(test_files), desc="Evaluating"):
        audio_path = os.path.join(data_path, row['audio_filename'])
        midi_path = os.path.join(data_path, row['midi_filename'])

        tsv_path = midi_path.rsplit('.', 1)[0] + '.tsv'
        if not os.path.exists(tsv_path):
            continue

        gt_data = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        if len(gt_data) == 0:
            continue
            
        gt_onsets = gt_data[:, 0]
        gt_keys = gt_data[:, 2]
        gt_vels = gt_data[:, 3] / 127.0

        try:
            audio, _ = librosa.load(audio_path, sr=OV_SAMPLE_RATE, mono=True)
            audio = torch.FloatTensor(audio).to(DEVICE)
            audio = audio / (audio.abs().max() + 1e-8)

            with torch.no_grad():
                mel = mel_extractor(audio)
                mel = mel.reshape(1, OV_N_MELS, -1)

                onset_pred, vel_pred = strided_inference(
                    model_inference, mel, CHUNK_SIZE, CHUNK_OVERLAP)

                onset_pred = onset_pred.to(DEVICE)
                vel_pred = vel_pred.to(DEVICE)

                df_preds = decoder(onset_pred, vel_pred, INFERENCE_THRESHOLD)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

        gt_df = pd.DataFrame({
            "onset": gt_onsets,
            "key": gt_keys,
            "vel": gt_vels
        })

        prf1, prf1_v = threshold_eval_single_file(
            gt_df, df_preds, SECS_PER_FRAME, MIN_MIDI,
            thresh=INFERENCE_THRESHOLD, shift_preds=0,
            tol_secs=0.05, tol_vel=0.1)

        metrics['metric/note/precision'].append(prf1[0])
        metrics['metric/note/recall'].append(prf1[1])
        metrics['metric/note/f1'].append(prf1[2])

        metrics['metric/note-with-velocity/precision'].append(prf1_v[0])
        metrics['metric/note-with-velocity/recall'].append(prf1_v[1])
        metrics['metric/note-with-velocity/f1'].append(prf1_v[2])

    print("\n--- Test Dataset Evaluation Results ---")
    
    table_data = {}
    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')
            if category not in table_data:
                table_data[category] = {}
            table_data[category][name] = np.mean(values)
            
    print("-" * 70)
    print(f"{'Metric':<30} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 70)

    rows = [
        ('note', 'Onset'),
        ('note-with-velocity', 'Onset + Velocity')
    ]

    table_cell_text = []
    table_row_labels = []

    for cat_key, display_name in rows:
        if cat_key in table_data:
            p = table_data[cat_key].get('precision', 0.0)
            r = table_data[cat_key].get('recall', 0.0)
            f1 = table_data[cat_key].get('f1', 0.0)
            print(f"{display_name:<30} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f}")
            table_row_labels.append(display_name)
            table_cell_text.append([f'{p:.4f}', f'{r:.4f}', f'{f1:.4f}'])
    print("-" * 70)

    if save_path and table_cell_text:
        fig, ax = plt.subplots(figsize=(8, 2.0))
        ax.axis('tight')
        ax.axis('off')

        col_labels = ['Precision', 'Recall', 'F1 Score']

        the_table = ax.table(cellText=table_cell_text,
                             rowLabels=table_row_labels,
                             colLabels=col_labels,
                             loc='center',
                             cellLoc='center')

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.5)

        image_path = os.path.join(save_path, 'evaluation_results-maestro.png')
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        print(f"\nEvaluation table saved to {image_path}")
        plt.close(fig)

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(
        model_dir,
        'OnsetsAndVelocities-maestro-260424-164512-epoch=10.pt')

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("--- OnsetsAndVelocities Evaluation ---")
    evaluate(MODEL_PATH, save_path=results_dir)