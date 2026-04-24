import os
import random
import pandas as pd
import numpy as np

from preprocessing.midi import parse_midi
from preprocessing.dataset import MAESTRO, MAPS
from preprocessing.constants import *
from preprocessing.visualize import visualizeFile

def processRandomFile(dataset_root, dataset):
    csv_path = os.path.join(dataset_root, 'maestro-v3.0.0.csv')

    if dataset.lower() == "maestro" and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df.sample(1).iloc[0]
        
        audio_path = os.path.join(dataset_root, row['audio_filename'])
        midi_path = os.path.join(dataset_root, row['midi_filename'])

        ds = MAESTRO(path=dataset_root, groups=[], sequence_length=SEQUENCE_LENGTH)
    else:
        ds = MAPS(path=dataset_root, groups=[], sequence_length=SEQUENCE_LENGTH)
        all_files = []
        for g in ds.available_groups():
            all_files.extend(ds.files(g))
            
        if not all_files:
            raise FileNotFoundError(f"Could not find any files in {dataset_root}. Make sure it's a valid MAESTRO or MAPS dataset.")
            
        audio_path, midi_path = random.choice(all_files)
    
    raw_data = ds.load(audio_path, midi_path)
    
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
    dataset_choice = "maestro"  # Change to "maps" to pick from MAPS_DATA_PATH
    path_to_check = DATA_PATH if dataset_choice == "maestro" else MAPS_DATA_PATH

    if os.path.exists(path_to_check):
        processRandomFile(path_to_check, dataset_choice)
    else:
        print(f"Path not found: {path_to_check}")
        