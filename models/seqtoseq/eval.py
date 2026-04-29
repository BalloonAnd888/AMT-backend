import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz

from models.seqtoseq.seq2seqPianoTranscription import Seq2SeqPianoTranscription
from models.seqtoseq.embedding.tokenEmbedding import (
    NOTE_OFFSET, NOTE_TOKENS, VELOCITY_OFFSET, VELOCITY_TOKENS,
    TIME_OFFSET, TIME_TOKENS, EOS_TOKEN, PAD_TOKEN, VOCAB_SIZE
)
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX

BATCH_SIZE = 2
INPUT_DIM = 229  # Number of Mel bands
MAX_SEQ_LEN = 4096
MIN_MIDI = 21
TIME_STEP_S = 0.01  # 10 ms resolution

def collate_fn(batch):
    audio = torch.stack([b['audio'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    velocities = torch.stack([b['velocity'] for b in batch]) if 'velocity' in batch[0] else None
    
    maxvals = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    audio = audio / maxvals
    
    return {'audio': audio, 'label': labels, 'velocity': velocities}

def decode_tokens(tokens):
    """Parses a sequence of discrete token IDs into Note Event dictionaries."""
    events = []
    current_time_s = 0.0
    current_velocity = 64
    
    for tok in tokens:
        if tok == EOS_TOKEN or tok == PAD_TOKEN:
            break
        elif TIME_OFFSET <= tok < TIME_OFFSET + TIME_TOKENS:
            bin_idx = tok - TIME_OFFSET
            current_time_s = bin_idx * TIME_STEP_S
        elif VELOCITY_OFFSET <= tok < VELOCITY_OFFSET + VELOCITY_TOKENS:
            current_velocity = tok - VELOCITY_OFFSET
        elif NOTE_OFFSET <= tok < NOTE_OFFSET + NOTE_TOKENS:
            pitch = tok - NOTE_OFFSET
            events.append({'time': current_time_s, 'pitch': pitch, 'velocity': current_velocity})
            
    return events

def events_to_notes(events):
    """Pairs Note-On and Note-Off events into distinct valid notes."""
    active = {}  # pitch -> (onset_s, velocity)
    notes = []
    
    for ev in sorted(events, key=lambda x: x['time']):
        if ev['velocity'] == 0:  # Note-off
            if ev['pitch'] in active:
                onset_s, vel = active.pop(ev['pitch'])
                notes.append((onset_s, ev['time'], ev['pitch'], vel))
        else:  # Note-on
            if ev['pitch'] in active:
                onset_s, vel = active.pop(ev['pitch'])
                notes.append((onset_s, ev['time'], ev['pitch'], vel))
            active[ev['pitch']] = (ev['time'], ev['velocity'])
            
    # Close remaining notes assuming default duration
    for pitch, (onset_s, vel) in active.items():
        notes.append((onset_s, onset_s + 0.5, pitch, vel))
        
    return sorted(notes)

def greedy_decode(model, src, max_len=4096):
    """Autoregressive decoding loop implementing Seq2Seq transcription logic."""
    device = src.device
    B = src.size(0)
    
    with torch.no_grad():
        # Pre-compute Encoder representation
        enc_output = model.input_linear(src)
        enc_output = model.src_pos_encoding(enc_output)
        for layer in model.encoder_layers:
            enc_output = layer(enc_output, None)
        enc_output = model.enc_norm(enc_output)
        
        # Start with padded token (or BOS if configured)
        generated = torch.full((B, 1), PAD_TOKEN, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            dec_output = model.tgt_embedding(generated)
            dec_output = model.tgt_pos_encoding(dec_output)
            for layer in model.decoder_layers:
                dec_output = layer(dec_output, enc_output, None, tgt_mask)
            dec_output = model.dec_norm(dec_output)
            
            # Predict the next token
            logits = model.fc_out(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1)
            
            # Pad tokens out once EOS is reached
            next_token = torch.where(finished, torch.full_like(next_token, PAD_TOKEN), next_token)
            finished = finished | (next_token == EOS_TOKEN)
            
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            if finished.all():
                break
                
    return generated.cpu().tolist()

def evaluate(model_path, save_path=None):
    print(f"Evaluating model on {DEVICE}...")

    test_dataset = MAESTRO(path=DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2SeqPianoTranscription(
        input_dim=INPUT_DIM,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded model from {model_path}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()
    
    mel_extractor = MelSpectrogram(
        sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, n_mels=INPUT_DIM, fmin=MEL_FMIN, fmax=MEL_FMAX
    ).to(DEVICE)

    metrics = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            audio = batch['audio'].to(DEVICE)
            labels = batch['label']
            velocities = batch['velocity']

            mel = mel_extractor(audio).transpose(1, 2)
            pred_tokens = greedy_decode(model, mel, max_len=MAX_SEQ_LEN)
            
            for i in range(audio.size(0)):
                T_steps = labels.size(1)
                gt_events = []
                for pitch_idx in range(labels.size(2)):
                    pitch = pitch_idx + MIN_MIDI
                    in_note = False
                    for t in range(T_steps):
                        state = labels[i, t, pitch_idx].item()
                        if state == 3 and not in_note:
                            in_note = True
                            onset_time = t * HOP_LENGTH / SAMPLE_RATE
                            v = int(velocities[i, t, pitch_idx].item() * 128.0) if velocities is not None else 64
                            v = max(1, min(127, v))
                            gt_events.append({'time': onset_time, 'pitch': pitch, 'velocity': v})
                        elif (state == 1 or state == 0) and in_note:
                            in_note = False
                            offset_time = t * HOP_LENGTH / SAMPLE_RATE
                            gt_events.append({'time': offset_time, 'pitch': pitch, 'velocity': 0})
                    if in_note:
                        offset_time = T_steps * HOP_LENGTH / SAMPLE_RATE
                        gt_events.append({'time': offset_time, 'pitch': pitch, 'velocity': 0})
                
                ref_notes = events_to_notes(gt_events)
                est_note_events = decode_tokens(pred_tokens[i])
                est_notes = events_to_notes(est_note_events)
                
                if len(ref_notes) == 0:
                    continue
                    
                i_ref = np.array([[n[0], n[1]] for n in ref_notes])
                p_ref = np.array([midi_to_hz(n[2]) for n in ref_notes])
                v_ref = np.array([n[3] for n in ref_notes]) / 127.0
                
                if len(est_notes) == 0:
                    i_est, p_est, v_est = np.empty((0, 2)), np.empty(0), np.empty(0)
                else:
                    i_est = np.array([[n[0], n[1]] for n in est_notes])
                    p_est = np.array([midi_to_hz(n[2]) for n in est_notes])
                    v_est = np.array([n[3] for n in est_notes]) / 127.0
                
                p, r, f, _ = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
                metrics['metric/note/f1'].append(f)
                
                p, r, f, _ = evaluate_notes(i_ref, p_ref, i_est, p_est)
                metrics['metric/note-with-offsets/f1'].append(f)
                
                p, r, f, _ = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
                metrics['metric/note-with-offsets-and-velocity/f1'].append(f)

    print("\n--- Seq2Seq MAESTRO Evaluation Results ---")
    print(f"Onset F1:                        {np.mean(metrics['metric/note/f1']):.4f}")
    print(f"Onset & Offset F1:               {np.mean(metrics['metric/note-with-offsets/f1']):.4f}")
    print(f"Onset, Offset, & Velocity F1:    {np.mean(metrics['metric/note-with-offsets-and-velocity/f1']):.4f}")

if __name__ == "__main__":
    # Update with your desired checkpoint path
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "seq2seq_20260429_013653_step100.pt")
    evaluate(MODEL_PATH)