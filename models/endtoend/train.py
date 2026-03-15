from datetime import datetime
import os
import numpy as np
import torch 
import random
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from tqdm import tqdm

from models.endtoend.endtoend import ETE
from models.endtoend.visualize import plot_learning_curves
from models.utils.constants import DEVICE
from models.utils.utils import visualize_data
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

BATCH_SIZE = 8
EPOCHS = 1
POS_WEIGHT = 10.0
MODEL_SAVE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GRAPH_SAVE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

def train():
    train_dataset = MAESTRO(path=DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH)
    validation_dataset = MAESTRO(path=DATA_PATH, groups=['validation'], sequence_length=SEQUENCE_LENGTH)

    # train_dataset = Subset(train_dataset, range(8))

    print(f"\ntrain_dataset length: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()
    
    print(f"validation_dataset length: {len(validation_dataset)}")
    if len(validation_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
    print(f"Length of validation_dataloader: {len(validation_dataloader)} batches of {BATCH_SIZE}...")

    # random_idx = random.randint(0, len(train_dataset) - 1)
    # visualize_data(train_dataset[random_idx])
    
    model = ETE(input_shape=1,
                output_shape=N_KEYS
                ).to(DEVICE)

    mel_extractor = MelSpectrogram().to(DEVICE)

    # batch = next(iter(train_dataloader))
    # audio = batch['audio'].to(DEVICE)
    # mel = mel_extractor(audio).unsqueeze(1)
    # print(mel.shape)
    # summary(model, input_data=mel)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    onset_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([POS_WEIGHT]).to(DEVICE))

    # print(model)
    results = {"train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            audio = batch['audio'].to(DEVICE)
            onset_ground_truth = batch['onset'].to(DEVICE).float()

            full_mels = mel_extractor(audio).unsqueeze(1)
            
            window_size = full_mels.shape[-1]
            half_window = window_size // 2
            padded_mels = torch.nn.functional.pad(full_mels, (half_window, half_window), mode='constant', value=full_mels.min())

            audio_np = audio.cpu().numpy()
            
            all_mels_for_model = []
            all_targets = []
            
            # Iterate over each sample in the batch
            for i in range(audio_np.shape[0]):
                # Detect onsets
                onset_frames = librosa.onset.onset_detect(
                    y=audio_np[i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units='frames'
                )
                
                if len(onset_frames) == 0:
                    onset_env = librosa.onset.onset_strength(y=audio_np[i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
                    onset_frames = [np.argmax(onset_env)]

                # For every detected onset, extract the mel window and target
                for t in onset_frames:
                    if t >= onset_ground_truth.shape[1]:
                        continue
                    
                    # Extract window centered at t
                    mel_window = padded_mels[i, :, :, t : t + window_size]
                    all_mels_for_model.append(mel_window)
                    all_targets.append(onset_ground_truth[i, t, :])

            chunk_size = 8
            batch_loss_accum = 0.0
            batch_acc_accum = 0.0
            num_chunks = 0

            for k in range(0, len(all_mels_for_model), chunk_size):
                chunk_mels = all_mels_for_model[k : k + chunk_size]
                chunk_targets = all_targets[k : k + chunk_size]

                # BatchNorm requires > 1 sample
                if len(chunk_mels) < 2:
                    continue

                mels_tensor = torch.stack(chunk_mels)
                targets_tensor = torch.stack(chunk_targets)

                optimizer.zero_grad()
                onset_pred = model(mels_tensor)
                loss = onset_loss_fn(onset_pred, targets_tensor)
                loss.backward()
                optimizer.step()
                
                pred_binary = (torch.sigmoid(onset_pred) > 0.5).float()
                acc = (pred_binary == targets_tensor).float().mean()

                batch_loss_accum += loss.item()
                batch_acc_accum += acc.item()
                num_chunks += 1
            
            if num_chunks > 0:
                train_loss += (batch_loss_accum / num_chunks)
                train_acc += (batch_acc_accum / num_chunks)

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        print(f"\n[Epoch {epoch} Summary]")
        print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2%}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.inference_mode():
            for batch in validation_dataloader:
                audio = batch['audio'].to(DEVICE)
                onset_ground_truth = batch['onset'].to(DEVICE).float()
                
                full_mels = mel_extractor(audio).unsqueeze(1)
                window_size = full_mels.shape[-1]
                half_window = window_size // 2
                padded_mels = torch.nn.functional.pad(full_mels, (half_window, half_window), mode='constant', value=full_mels.min())

                audio_np = audio.cpu().numpy()
                
                all_mels_for_model = []
                all_targets = []

                for i in range(audio_np.shape[0]):
                    onset_frames = librosa.onset.onset_detect(
                        y=audio_np[i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, units='frames'
                    )
                    
                    if len(onset_frames) == 0:
                        onset_env = librosa.onset.onset_strength(y=audio_np[i], sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
                        onset_frames = [np.argmax(onset_env)]

                    for t in onset_frames:
                        if t >= onset_ground_truth.shape[1]:
                            continue
                        
                        mel_window = padded_mels[i, :, :, t : t + window_size]
                        all_mels_for_model.append(mel_window)
                        all_targets.append(onset_ground_truth[i, t, :])

                # Process in chunks
                chunk_size = 8
                batch_loss_accum = 0.0
                batch_acc_accum = 0.0
                num_chunks = 0

                for k in range(0, len(all_mels_for_model), chunk_size):
                    chunk_mels = all_mels_for_model[k : k + chunk_size]
                    chunk_targets = all_targets[k : k + chunk_size]

                    mels_tensor = torch.stack(chunk_mels)
                    targets_tensor = torch.stack(chunk_targets)
                    
                    val_onset_pred = model(mels_tensor)
                    loss = onset_loss_fn(val_onset_pred, targets_tensor)
                    pred_binary = (torch.sigmoid(val_onset_pred) > 0.5).float()
                    acc = (pred_binary == targets_tensor).float().mean()

                    batch_loss_accum += loss.item()
                    batch_acc_accum += acc.item()
                    num_chunks += 1

                if num_chunks > 0:
                    val_loss += (batch_loss_accum / num_chunks)
                    val_acc += (batch_acc_accum / num_chunks)

            val_loss /= len(validation_dataloader)
            val_acc /= len(validation_dataloader)

            print(f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2%}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"ete_model_{timestamp}.pt"))

    plot_learning_curves(results, timestamp, save_dir=GRAPH_SAVE_DIR)
    
if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    train()
