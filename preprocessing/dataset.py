import os 
from abc import abstractmethod 
from glob import glob 

import numpy as np 
import librosa 
import torch 
import pandas as pd 
from torch.utils.data import Dataset 
from tqdm import tqdm 

from preprocessing.constants import * 
from preprocessing.midi import parse_midi 

class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEVICE):
        self.path = path 
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed) 

        self.data = []
        print(f"Loading {len(self.groups)} group{'s' if len(self.groups) > 1 else ""} "
              f"of {self.__class__.__name__} at {path}")
        
        for group in self.groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))
    
    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])

            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH 
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps 

            begin = step_begin * HOP_LENGTH 
            end = begin + self.sequence_length 

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device) 
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float() 
        
        result['audio'] = result['audio'].float().div_(32768)

        result['onset'] = (result['label'] == 3).float() 
        result['offset'] = (result['label'] == 1).float() 
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result 

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError 
    
    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError 

    def load(self, audio_path, tsv_path):
        saved_data_path = audio_path.rsplit('.', 1)[0] + '.pt'

        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)
        
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            return dict(path=audio_path, audio=torch.zeros(1), label=torch.zeros(1, 88), velocity=torch.zeros(1, 88))
        
        audio = (audio * 32768).astype(np.int16)
        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, N_KEYS, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, N_KEYS, dtype=torch.uint8)

        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            if 0 <= f < N_KEYS:
                label[left:onset_right, f] = 3
                label[onset_right:frame_right, f] = 2
                label[frame_right:offset_right, f] = 1
                velocity[left:frame_right, f] = vel 
        
        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)

        torch.save(data, saved_data_path)
        return data 
    
class MAESTRO(PianoRollAudioDataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)
    
    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']
    
    def files(self, group):
        if group not in self.available_groups():
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))
            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f"Group {group} is empty")
        else:
            csv_path = os.path.join(self.path, "maestro-v3.0.0.csv")

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Could not find maestro csv at {self.path}")
            
            df = pd.read_csv(csv_path) 

            filtered_df = df[df['split'] == group]

            files = []
            for _, row in filtered_df.iterrows():
                audio_file = os.path.join(self.path, row['audio_filename'])
                midi_file = os.path.join(self.path, row['midi_filename'])
                files.append((audio_file, midi_file))

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.rsplit('.', 1)[0] + '.tsv'

            if not os.path.exists(tsv_filename):
                midi_data = parse_midi(midi_path)
                midi_array = np.array([[n.start, n.end, n.pitch, n.velocity] for n in midi_data])
                np.savetxt(tsv_filename, midi_array, fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')

            result.append((audio_path, tsv_filename))
        
        return result 
