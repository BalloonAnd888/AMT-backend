import torch
import os
import numpy as np
import librosa

from torch.utils.data import DataLoader
from tqdm import tqdm
from models.endtoend.endtoend import ETE
from models.utils.constants import DEVICE
from preprocessing.constants import DATA_PATH, N_KEYS, SEQUENCE_LENGTH, SAMPLE_RATE, HOP_LENGTH
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

POS_WEIGHT = 10.0

def evaluate(model_path, data_path, batch_size):
    print(f"Loading model from: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {DEVICE}")

    model = ETE(
        input_shape=1,
        output_shape=N_KEYS
    ).to(DEVICE)

    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    # 2. Data Loading
    test_dataset = MAESTRO(path=data_path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
        
    print(f"test_dataset length: {len(test_dataset)}\n")
    if len(test_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Batches: {len(test_dataloader)}")

    mel_extractor = MelSpectrogram().to(DEVICE)

    onset_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([POS_WEIGHT]).to(DEVICE))

    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.inference_mode():
        progress_bar = tqdm(test_dataloader, desc="Testing")
        for batch in progress_bar:
            audio = batch['audio'].to(DEVICE)
            onset_ground_truth = batch['onset'].to(DEVICE).float()

            mel = mel_extractor(audio).unsqueeze(1)
            
            window_size = mel.shape[-1]
            half_window = window_size // 2
            padded_mels = torch.nn.functional.pad(mel, (half_window, half_window), mode='constant', value=mel.min())

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

            chunk_size = 8
            batch_loss_accum = 0.0
            batch_acc_accum = 0.0
            num_chunks = 0

            for k in range(0, len(all_mels_for_model), chunk_size):
                chunk_mels = all_mels_for_model[k : k + chunk_size]
                chunk_targets = all_targets[k : k + chunk_size]

                mels_tensor = torch.stack(chunk_mels)
                targets_tensor = torch.stack(chunk_targets)
                
                test_onset_pred = model(mels_tensor)
                loss = onset_loss_fn(test_onset_pred, targets_tensor)
                pred_binary = (torch.sigmoid(test_onset_pred) > 0.5).float()
                acc = (pred_binary == targets_tensor).float().mean()

                batch_loss_accum += loss.item()
                batch_acc_accum += acc.item()
                num_chunks += 1

            if num_chunks > 0:
                test_loss += (batch_loss_accum / num_chunks)
                test_acc += (batch_acc_accum / num_chunks)

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"\nTest Loss: {test_loss:.5f} | Test Acc: {test_acc:.2%}")

if __name__ == "__main__":
    # Default to the models directory relative to this script
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'ete_model_20260203_195050.pt')
    evaluate(MODEL_PATH, DATA_PATH, batch_size=8)
