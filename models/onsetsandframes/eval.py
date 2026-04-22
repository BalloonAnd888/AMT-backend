import os
import sys
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import hmean
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz

from models.onsetsandframes.of import OnsetsAndFrames
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, N_MELS, N_KEYS, SEQUENCE_LENGTH, HOP_LENGTH, SAMPLE_RATE, MIN_MIDI
from preprocessing.dataset import MAESTRO
from models.onsetsandframes.decoding import extract_notes, notes_to_frames

BATCH_SIZE = 8
eps = sys.float_info.epsilon

def evaluate(model_path, onset_threshold=0.5, frame_threshold=0.5, save_path=None):
    print(f"Evaluating model on {DEVICE}...")

    # Load test dataset
    test_dataset = MAESTRO(path=DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    if len(test_dataset) == 0:
        print("ERROR: Test dataset is empty.")
        return
        
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = OnsetsAndFrames(
        input_features=N_MELS,
        output_features=N_KEYS,
        model_complexity=48
    ).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded model from {model_path}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()

    metrics = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(DEVICE)
                    if key in ['onset', 'offset', 'frame', 'velocity']:
                        batch[key] = batch[key].float()
            
            pred, losses = model.run_on_batch(batch)
            
            for key, loss in losses.items():
                metrics[key].append(loss.item())
            
            for key, value in pred.items():
                value.relu_()
                
            batch_size = batch['audio'].size(0)
            for i in range(batch_size):
                p_ref, i_ref, v_ref = extract_notes(batch['onset'][i], batch['frame'][i], batch['velocity'][i])
                p_est, i_est, v_est = extract_notes(pred['onset'][i], pred['frame'][i], pred['velocity'][i], onset_threshold, frame_threshold)

                t_ref, f_ref = notes_to_frames(p_ref, i_ref, batch['frame'][i].shape)
                t_est, f_est = notes_to_frames(p_est, i_est, pred['frame'][i].shape)

                scaling = HOP_LENGTH / SAMPLE_RATE

                i_ref = (i_ref * scaling).reshape(-1, 2)
                p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
                i_est = (i_est * scaling).reshape(-1, 2)
                p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

                t_ref = t_ref.astype(np.float64) * scaling
                f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
                t_est = t_est.astype(np.float64) * scaling
                f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

                p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
                metrics['metric/note/precision'].append(p)
                metrics['metric/note/recall'].append(r)
                metrics['metric/note/f1'].append(f)
                metrics['metric/note/overlap'].append(o)

                p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
                metrics['metric/note-with-offsets/precision'].append(p)
                metrics['metric/note-with-offsets/recall'].append(r)
                metrics['metric/note-with-offsets/f1'].append(f)
                metrics['metric/note-with-offsets/overlap'].append(o)

                p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                          offset_ratio=None, velocity_tolerance=0.1)
                metrics['metric/note-with-velocity/precision'].append(p)
                metrics['metric/note-with-velocity/recall'].append(r)
                metrics['metric/note-with-velocity/f1'].append(f)
                metrics['metric/note-with-velocity/overlap'].append(o)

                p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
                metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
                metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
                metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
                metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

                frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
                metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

                for key, loss in frame_metrics.items():
                    metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

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
        ('frame', 'Frame'),
        ('note', 'Note'),
        ('note-with-offsets', 'Note w/ offset'),
        ('note-with-offsets-and-velocity', 'Note w/ offset & velocity')
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
        fig, ax = plt.subplots(figsize=(8, 2.5))
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

        image_path = os.path.join(save_path, 'evaluation_results.png')
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        print(f"\nEvaluation table saved to {image_path}")
        plt.close(fig)

if __name__ == "__main__":
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "modelFiles")
    MODEL_PATH = os.path.join(MODEL_DIR, "onsetsandframes-260209-204911-5000.pt")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    evaluate(MODEL_PATH, save_path=results_dir)
