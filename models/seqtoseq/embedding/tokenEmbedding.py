import math
import torch
from torch import nn

# ── Vocabulary Constants ──────────────────────────────────────────────────
NOTE_OFFSET = 0
NOTE_TOKENS = 128
VELOCITY_OFFSET = 128
VELOCITY_TOKENS = 128
TIME_OFFSET = 256
TIME_TOKENS = 6000 # 60 seconds at 10ms resolution
EOS_TOKEN = 6256
VOCAB_SIZE = 6257
PAD_TOKEN = 0

class Tokenizer:
    """
    Converts continuous piano roll representations into discrete sequences 
    of Time, Note, and Velocity tokens for the Transformer to model.
    """
    def __init__(self, sample_rate=16000, hop_length=128, min_midi=21, max_seq_len=4096):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.min_midi = min_midi
        self.max_seq_len = max_seq_len
        self.time_step_s = 0.01  # 10 ms bins

    def encode_events(self, events):
        """
        Convert a list of parsed note events into discrete tokens.
        """
        # Sort by time, then pitch, then velocity (note-offs before note-ons if same time)
        events.sort(key=lambda x: (x['time'], x['pitch'], x['velocity']))
        
        tokens = []
        for event in events:
            # 1. Time Token
            bin_idx = int(event['time'] / self.time_step_s)
            bin_idx = max(0, min(bin_idx, TIME_TOKENS - 1))
            time_token = TIME_OFFSET + bin_idx
            
            # 2. Note Token
            pitch = event['pitch']
            note_token = NOTE_OFFSET + max(0, min(pitch, NOTE_TOKENS - 1))
            
            # 3. Velocity Token
            velocity = event['velocity']
            vel_token = VELOCITY_OFFSET + max(0, min(velocity, VELOCITY_TOKENS - 1))
            
            # Append triplet
            tokens.extend([time_token, note_token, vel_token])

        return tokens

    def tokenize(self, labels: torch.Tensor, velocities: torch.Tensor = None) -> torch.Tensor:
        """
        Converts a batch of dense piano roll targets into token sequences.
        labels: (B, T_steps, N_KEYS) - categorical/binary labels (0=off, 1=offset, 3=onset)
        velocities: (B, T_steps, N_KEYS) - scaled [0, 1]. Defaults to 64 if None.
        Returns padded token tensor: (B, max_seq_len)
        """
        B, T_steps, N_KEYS = labels.shape
        batch_tokens = []

        for b in range(B):
            events = []
            for pitch_idx in range(N_KEYS):
                pitch = pitch_idx + self.min_midi
                in_note = False
                
                for t in range(T_steps):
                    state = labels[b, t, pitch_idx].item()
                    
                    # State == 3 means onset
                    if state == 3 and not in_note:
                        in_note = True
                        onset_time = t * self.hop_length / self.sample_rate
                        if velocities is not None:
                            v = int(velocities[b, t, pitch_idx].item() * 128.0)
                            v = max(1, min(127, v))
                        else:
                            v = 64 # Default velocity if not provided
                        events.append({'time': onset_time, 'pitch': pitch, 'velocity': v})
                        
                    # State == 1 means offset, State == 0 means off
                    elif (state == 1 or state == 0) and in_note:
                        in_note = False
                        offset_time = t * self.hop_length / self.sample_rate
                        events.append({'time': offset_time, 'pitch': pitch, 'velocity': 0})
                
                # Close notes that carry over the edge of the clip
                if in_note:
                    offset_time = T_steps * self.hop_length / self.sample_rate
                    events.append({'time': offset_time, 'pitch': pitch, 'velocity': 0})
                    
            tokens = self.encode_events(events)
            
            # Truncate keeping room for EOS
            if len(tokens) > self.max_seq_len - 1:
                tokens = tokens[:self.max_seq_len - 1]
            tokens.append(EOS_TOKEN)
                
            batch_tokens.append(torch.tensor(tokens, dtype=torch.long))

        # Pad sequences (using PAD_TOKEN = 0)
        padded_tokens = torch.full((B, self.max_seq_len), PAD_TOKEN, dtype=torch.long)
        for b, t_tensor in enumerate(batch_tokens):
            padded_tokens[b, :len(t_tensor)] = t_tensor

        return padded_tokens

class TokenEmbedding(nn.Module):
    """
    Embeds the discrete tokens into the dense dimensional space of the model.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale the embeddings by sqrt(d_model) as described in 'Attention Is All You Need'
        return self.embedding(x) * math.sqrt(self.d_model)