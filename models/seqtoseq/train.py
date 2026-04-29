import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from models.seqtoseq.seq2seqPianoTranscription import Seq2SeqPianoTranscription
from models.seqtoseq.embedding.tokenEmbedding import Tokenizer, VOCAB_SIZE
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX

BATCH_SIZE = 8
TRAIN_STEPS = 400
VAL_EVERY = 100
INPUT_DIM = 229  # Number of Mel bands
MAX_SEQ_LEN = 4096
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

def collate_fn(batch):
    audio = torch.stack([b['audio'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    
    if 'velocity' in batch[0]:
        velocities = torch.stack([b['velocity'] for b in batch])
    else:
        velocities = None

    # Peak normalize audio
    maxvals = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    audio = audio / maxvals
    
    return {'audio': audio, 'label': labels, 'velocity': velocities}

def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    print("Loading datasets...")
    train_dataset = MAESTRO(path=DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH)
    val_dataset = MAESTRO(path=DATA_PATH, groups=['validation'], sequence_length=SEQUENCE_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Device: {DEVICE}")
    print("Initializing model...")
    model = Seq2SeqPianoTranscription(
        input_dim=INPUT_DIM,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)

    mel_extractor = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        n_mels=INPUT_DIM,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-4)

    tokenizer = Tokenizer(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        max_seq_len=MAX_SEQ_LEN
    )

    print("Starting training...")
    step = 0
    train_iter = iter(train_dataloader)
    train_loss = 0.0
    model.train()
        
    with tqdm(total=TRAIN_STEPS, desc="Training") as progress_bar:
        while step < TRAIN_STEPS:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
                
            audio = batch['audio'].to(DEVICE)
            
            with torch.no_grad():
                # Extract Mel Spectrogram
                mel = mel_extractor(audio) # Shape: (B, INPUT_DIM, T)
                mel = mel.transpose(1, 2) # Shape: (B, T, INPUT_DIM)
            
            # Tokenize batch['label'] into discrete target tokens
            target_tokens = tokenizer.tokenize(batch['label'], batch['velocity']).to(DEVICE)
            
            tgt_input = target_tokens[:, :-1]
            tgt_target = target_tokens[:, 1:]

            optimizer.zero_grad()
            
            # Forward pass
            output = model(mel, tgt_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_target.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            progress_bar.update(1)
            step += 1
            
            if step % VAL_EVERY == 0 or step == TRAIN_STEPS:
                avg_train_loss = train_loss / VAL_EVERY
                train_loss = 0.0
                
                # Validation Loop
                model.eval()
                val_loss = 0.0
                val_progress_bar = tqdm(val_dataloader, desc=f"Step {step} [Val]", leave=False)
                
                with torch.no_grad():
                    for val_batch in val_progress_bar:
                        val_audio = val_batch['audio'].to(DEVICE)
                        val_mel = mel_extractor(val_audio).transpose(1, 2)
                        
                        # Tokenize validation batch labels
                        val_target_tokens = tokenizer.tokenize(val_batch['label'], val_batch['velocity']).to(DEVICE)
                        val_tgt_input = val_target_tokens[:, :-1]
                        val_tgt_target = val_target_tokens[:, 1:]
                        
                        val_output = model(val_mel, val_tgt_input)
                        v_loss = criterion(val_output.reshape(-1, VOCAB_SIZE), val_tgt_target.reshape(-1))
                        
                        val_loss += v_loss.item()
                        val_progress_bar.set_postfix(loss=f"{v_loss.item():.4f}")
                        
                avg_val_loss = val_loss / len(val_dataloader)
                tqdm.write(f"\nStep {step} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
                
                # Save model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(MODEL_SAVE_DIR, f"seq2seq_{timestamp}_step{step}.pt")
                torch.save(model.state_dict(), save_path)
                
                model.train()

if __name__ == "__main__":
    train()
