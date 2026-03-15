from datetime import datetime
import os
import numpy as np
import torch 
import random
import matplotlib.pyplot as plt
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
            # Original shape: [batch, 88, time]
            onset = batch['onset'].to(DEVICE).float() 
            
            # 1. Detect onsets for each audio sample in the batch
            # Moving to CPU/NumPy for librosa processing
            audio_np = audio.cpu().numpy()
            
            # We will store the "best" target frame for each item in the batch
            target_onsets = []
            
            for i in range(audio_np.shape[0]):
                # Calculate onset strength envelope (Spectral Flux) 
                onset_env = librosa.onset.onset_strength(
                    y=audio_np[i], 
                    sr=SAMPLE_RATE, 
                    hop_length=HOP_LENGTH
                )
                
                # Find the frame index with the maximum onset strength in this window
                # This focuses the model on the "attack phase"
                best_onset_frame = np.argmax(onset_env)
                best_onset_frame = min(best_onset_frame, onset.size(1) - 1)
                target_onsets.append(best_onset_frame)

            # 2. Extract features and align labels
            mel = mel_extractor(audio).unsqueeze(1)
            onset_pred = model(mel)

            # Instead of onset[:, onset.size(1) // 2, :], 
            # we select the frame identified by our onset analysis
            batch_indices = torch.arange(audio.size(0)).to(DEVICE)
            target_frame_indices = torch.tensor(target_onsets).to(DEVICE)
            
            # Align labels with the detected transient moment
            onset_targets = onset[batch_indices, target_frame_indices, :]

            # 3. Standard Optimization
            loss = onset_loss_fn(onset_pred, onset_targets)
            
            # Calculate accuracy
            pred_binary = (torch.sigmoid(onset_pred) > 0.5).float()
            acc = (pred_binary == onset_targets).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()

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
                val_onset = batch['onset'].to(DEVICE).float()
                
                # 1. Detect onsets for each audio sample in the batch (Validation)
                audio_np = audio.cpu().numpy()
                target_onsets = []
                for i in range(audio_np.shape[0]):
                    onset_env = librosa.onset.onset_strength(
                        y=audio_np[i], 
                        sr=SAMPLE_RATE, 
                        hop_length=HOP_LENGTH
                    )
                    best_onset_frame = np.argmax(onset_env)
                    best_onset_frame = min(best_onset_frame, val_onset.size(1) - 1)
                    target_onsets.append(best_onset_frame)

                batch_indices = torch.arange(audio.size(0)).to(DEVICE)
                target_frame_indices = torch.tensor(target_onsets).to(DEVICE)
                val_onset_targets = val_onset[batch_indices, target_frame_indices, :]

                mel = mel_extractor(audio).unsqueeze(1)
                
                val_onset_pred = model(mel)
                loss = onset_loss_fn(val_onset_pred, val_onset_targets)
                pred_binary = (torch.sigmoid(val_onset_pred) > 0.5).float()
                acc = (pred_binary == val_onset_targets).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()

            val_loss /= len(validation_dataloader)
            val_acc /= len(validation_dataloader)

            print(f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2%}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"ov_model_{timestamp}.pt"))

    # plot_learning_curves(results, timestamp, save_dir=GRAPH_SAVE_DIR)
    
if __name__ == "__main__":
    torch.set_printoptions(profile="full")
    train()
