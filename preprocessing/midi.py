import numpy as np
import pretty_midi
from preprocessing.constants import *

def parse_midi(path):
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(f"Error loading MIDI {path}: {e}")
        return []
    
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append(note)
    return notes

def compute_targets(notes, start_sample, duration_samples):
    start_time = start_sample / SAMPLE_RATE
    end_time = (start_sample * duration_samples) / SAMPLE_RATE

    n_frames = duration_samples // HOP_LENGTH

    onsets = np.zeros((n_frames, N_KEYS), dtype=np.float32)
    offsets = np.zeros((n_frames, N_KEYS), dtype=np.float32)
    frames = np.zeros((n_frames, N_KEYS), dtype=np.float32)
    velocities = np.zeros((n_frames, N_KEYS), dtype=np.float32)

    for note in notes:
        if note.end < start_time or note.start > end_time:
            continue

        rel_start = note.start - start_time
        rel_end = note.end - start_time

        start_frame = int(rel_start * SAMPLE_RATE / HOP_LENGTH)
        end_frame = int(rel_end * SAMPLE_RATE / HOP_LENGTH) + 1

        pitch_idx = note.pitch - MIN_MIDI

        if pitch_idx < 0 or pitch_idx >= N_KEYS:
            continue 
        
        frame_start_clamped = max(0, start_frame)
        frame_end_clamped = min(n_frames, end_frame)

        frames[frame_start_clamped:frame_end_clamped, pitch_idx] = 1

        velocities[frame_start_clamped:frame_end_clamped, pitch_idx] = note.velocity / 127.0

        if 0 <= start_frame < n_frames:
            onsets[start_frame, pitch_idx] = 1

            if HOPS_IN_ONSET > 1:
                win = HOPS_IN_ONSET // 2
                lower = max(0, start_frame - win)
                upper = min(n_frames, start_frame + win + 1)
                onsets[lower:upper, pitch_idx] = 1

        if 0 <= end_frame < n_frames:
            offsets[end_frame, pitch_idx] = 1

            if HOPS_IN_OFFSET > 1:
                win = HOPS_IN_OFFSET // 2
                lower = max(0, end_frame - win)
                upper = min(n_frames, end_frame + win + 1)
                offsets[lower:upper, pitch_idx] = 1
        
    return onsets, offsets, frames, velocities
