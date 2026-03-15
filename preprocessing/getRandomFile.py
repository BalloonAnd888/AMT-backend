import os
import pandas as pd
import numpy as np

from preprocessing.midi import parse_midi
from preprocessing.dataset import MAESTRO
from preprocessing.constants import *
from preprocessing.visualize import visualizeFile

def processRandomFile(dataset_root):
    csv_path = os.path.join(dataset_root, 'maestro-v3.0.0.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find metadata CSV in {dataset_root}")

    df = pd.read_csv(csv_path)
    row = df.sample(1).iloc[0]
    
    audio_path = os.path.join(dataset_root, row['audio_filename'])
    midi_path = os.path.join(dataset_root, row['midi_filename'])
    tsv_path = midi_path.rsplit('.', 1)[0] + '.tsv'
    
    if not os.path.exists(tsv_path):
        print(f"Generating TSV for {row['midi_filename']}...")
        try:
            midi_notes = parse_midi(midi_path)
            midi_list = [[n.start, n.end, n.pitch, n.velocity] for n in midi_notes]
            np.savetxt(tsv_path, np.array(midi_list), fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')
        except Exception as e:
            print(f"Error parsing MIDI: {e}")
            return

    ds = MAESTRO(path=dataset_root, groups=[], sequence_length=SEQUENCE_LENGTH)
    
    raw_data = ds.load(audio_path, tsv_path)
    
    audio_len = len(raw_data['audio'])
    
    if audio_len < SEQUENCE_LENGTH:
        begin = 0
        end = audio_len
        step_begin = 0
        step_end = audio_len // HOP_LENGTH
    else:
        step_begin = np.random.randint(audio_len - SEQUENCE_LENGTH) // HOP_LENGTH
        n_steps = SEQUENCE_LENGTH // HOP_LENGTH
        step_end = step_begin + n_steps
        begin = step_begin * HOP_LENGTH
        end = begin + SEQUENCE_LENGTH
    
    processed_data = {
        'path': audio_path,
        'audio': raw_data['audio'][begin:end].to(DEVICE).float().div_(32768.0),
        'label': raw_data['label'][step_begin:step_end, :].to(DEVICE)
    }

    visualizeFile(processed_data)

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        processRandomFile(DATA_PATH)
    else:
        print(f"Path not found: {DATA_PATH}")
        