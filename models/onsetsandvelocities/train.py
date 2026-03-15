#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from typing import List, Optional
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.onsetsandvelocities.optimizers import AdamWR
from preprocessing.constants import *
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.utils.utils import plot_learning_curves

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 10

# Phase Toggle: Set False and provide SNAPSHOT_INPATH to train only velocity
TRAINABLE_ONSETS: bool = True 
SNAPSHOT_INPATH: Optional[str] = None # e.g., "models/ov_model_onsets_only.pt"

CONV1X1_HEAD: List[int] = (200, 200)

# Optimizer Hyperparameters
LR_MAX: float = 0.008
LR_WARMUP: float = 0.5
LR_PERIOD: int = 1000
LR_DECAY: float = 0.975
LR_SLOWDOWN: float = 1.0
MOMENTUM: float = 0.95
WEIGHT_DECAY: float = 0.0003
BATCH_NORM: float = 0.95
DROPOUT: float = 0.15
LEAKY_RELU_SLOPE: Optional[float] = 0.1

# Loss Constants
ONSET_POSITIVES_WEIGHT: float = 8.0
VEL_LOSS_LAMBDA: float = 10.0 # Weight for velocity loss

IN_CHANS = 2 
MODEL_SAVE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
GRAPH_SAVE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# ##############################################################################
# LOSS FUNCTION
# ##############################################################################
class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        """
        """
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        #
        return result

# ##############################################################################
# TRAINING FUNCTION
# ##############################################################################
def train():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(f"Model will be saved to: {MODEL_SAVE_DIR}")
    print(f"Graph will be saved to: {GRAPH_SAVE_DIR}")
    print(f"Device: {DEVICE} | Training Onsets: {TRAINABLE_ONSETS}")

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_ons_loss": [], "val_ons_loss": [],
        "train_vel_loss": [], "val_vel_loss": [],
        "train_ons_acc": [], "val_ons_acc": [],
        "train_vel_acc": [], "val_vel_acc": []
    }

    # 1. Model Initialization
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    # Load pre-trained onset weights if starting Phase 2
    snapshot_path = SNAPSHOT_INPATH
    if snapshot_path and not TRAINABLE_ONSETS:
        model_weight_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        snapshot_path = os.path.join(model_weight_dir, snapshot_path)

        if os.path.exists(snapshot_path):
            print(f"Loading weights from {snapshot_path}...")
            model.load_state_dict(torch.load(snapshot_path, map_location=DEVICE))
        else:
            print(f"WARNING: Snapshot file not found at {snapshot_path}")
            exit()

    # 2. Optimizer & Loss Setup
    # Filter params: if not training onsets, only optimize the velocity head
    trainable_params = model.parameters() if TRAINABLE_ONSETS else model.velocity_stage.parameters()

    def model_saver(cycle=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"ov_checkpoint_{timestamp}.pt"))

    opt_hpars = {
        "lr_max": LR_MAX, "lr": LR_MAX,
        "lr_period": LR_PERIOD, "lr_decay": LR_DECAY,
        "lr_slowdown": LR_SLOWDOWN, "cycle_end_hook_fn": model_saver,
        "cycle_warmup": LR_WARMUP, "weight_decay": WEIGHT_DECAY,
        "betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}

    optimizer = AdamWR(trainable_params, **opt_hpars)

    ons_pos_weights = torch.FloatTensor([ONSET_POSITIVES_WEIGHT]).to(DEVICE)
    ons_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # 3. Data Loading
    train_dataset = MAESTRO(path=DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH)
    validation_dataset = MAESTRO(path=DATA_PATH, groups=['validation'], sequence_length=SEQUENCE_LENGTH)

    # Only look at 10 samples for now
    # train_dataset = Subset(train_dataset, range(50))

    print(f"\ntrain_dataset length: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()
    
    print(f"validation_dataset length: {len(validation_dataset)}")
    if len(validation_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Length of dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")

    mel_extractor = MelSpectrogram().to(DEVICE)

    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_ons_loss, train_vel_loss, train_ons_acc, train_vel_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_ons_loss, val_vel_loss, val_ons_acc, val_vel_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            # Move and Format Data
            audio = batch['audio'].to(DEVICE)
            # Targets: (Batch, Time, Keys) -> (Batch, Keys, Time-1)
            onset = batch['onset'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]
            velocity = batch['velocity'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]

            mel = mel_extractor(audio)

            # Forward Pass
            onset_stack, vels = model(mel)

            # Calculate loss
            ons_loss = sum(ons_loss_fn(p, (onset > 0).float()) for p in onset_stack) / len(onset_stack)
            
            vel_mask = (onset > 0).float()
            vel_target = velocity / 127.0
            vel_loss = vel_loss_fn(vels, vel_target, mask=vel_mask)
            
            if TRAINABLE_ONSETS:
                loss = ons_loss + (vel_loss * VEL_LOSS_LAMBDA)
            else:
                loss = vel_loss * VEL_LOSS_LAMBDA

            train_loss += loss.item()
            train_ons_loss += ons_loss.item()
            train_vel_loss += vel_loss.item()

            # Calculate accuracy
            pred_ons= (torch.sigmoid(onset_stack[-1]) > 0.5).float()
            ons_acc = (pred_ons == (onset > 0).float()).float().mean()

            pred_vels = torch.sigmoid(vels)
            if vel_mask.sum() > 0:
                active_err = torch.abs(pred_vels[vel_mask > 0] - vel_target[vel_mask > 0])
                vel_acc = (active_err < 0.1).float().mean()
            else:
                vel_acc = torch.tensor(0.0)

            train_ons_acc += ons_acc.item()
            train_vel_acc += vel_acc.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_ons_loss /= len(train_dataloader)
        train_vel_loss /= len(train_dataloader)
        train_ons_acc /= len(train_dataloader)
        train_vel_acc /= len(train_dataloader)

        history["train_loss"].append(train_loss)
        history["train_ons_loss"].append(train_ons_loss)
        history["train_vel_loss"].append(train_vel_loss)
        history["train_ons_acc"].append(train_ons_acc)
        history["train_vel_acc"].append(train_vel_acc)

        # Epoch Summary
        print(f"\n[Epoch {epoch} Summary]")
        print(f"Train Loss: {train_loss:.5f} | Onset Loss: {train_ons_loss:.5f} | Vel Loss: {train_vel_loss:.5f}")
        print(f"Train Accuracy: Onset {train_ons_acc:.2%}, Velocity {train_vel_acc:.2%}")
        
        # Validation
        model.eval()

        with torch.inference_mode():
            for batch in validation_dataloader:
                # Move and Format Data
                audio = batch['audio'].to(DEVICE)
                # Targets: (Batch, Time, Keys) -> (Batch, Keys, Time-1)
                onset = batch['onset'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]
                velocity = batch['velocity'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]

                mel = mel_extractor(audio)
                val_pred_onset_stack, val_pred_vels = model(mel)

                batch_ons_loss = sum(ons_loss_fn(p, (onset > 0).float()) for p in val_pred_onset_stack) / len(val_pred_onset_stack)
                
                vel_mask = (onset > 0).float()
                vel_target = velocity / 127.0
                batch_vel_loss = vel_loss_fn(val_pred_vels, vel_target, mask=vel_mask)

                if TRAINABLE_ONSETS:
                    loss = batch_ons_loss + (batch_vel_loss * VEL_LOSS_LAMBDA)
                else:
                    loss = batch_vel_loss * VEL_LOSS_LAMBDA

                val_loss += loss.item()
                val_ons_loss += batch_ons_loss.item()
                val_vel_loss += batch_vel_loss.item()

                val_pred_ons= (torch.sigmoid(val_pred_onset_stack[-1]) > 0.5).float()
                val_ons_acc += (val_pred_ons == (onset > 0).float()).float().mean().item()

                pred_vels = torch.sigmoid(val_pred_vels)
                if vel_mask.sum() > 0:
                    active_err = torch.abs(pred_vels[vel_mask > 0] - vel_target[vel_mask > 0])
                    val_vel_acc += (active_err < 0.1).float().mean().item()
                else:
                    val_vel_acc += 0.0

            val_loss /= len(validation_dataloader)
            val_ons_loss /= len(validation_dataloader)
            val_vel_loss /= len(validation_dataloader)
            val_ons_acc /= len(validation_dataloader)
            val_vel_acc /= len(validation_dataloader)

            history["val_loss"].append(val_loss)
            history["val_ons_loss"].append(val_ons_loss)
            history["val_vel_loss"].append(val_vel_loss)
            history["val_ons_acc"].append(val_ons_acc)
            history["val_vel_acc"].append(val_vel_acc)

            print(f"Val Loss: {val_loss:.5f} | Onset Loss: {val_ons_loss:.5f} | Vel Loss: {val_vel_loss:.5f}")
            print(f"Val Accuracy: Onset {val_ons_acc:.2%} | Velocity {val_vel_acc:.2%}")

    # Final Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"ov_model_{timestamp}.pt"))

    # Plot learning curves
    plot_learning_curves(history, timestamp, save_dir=GRAPH_SAVE_DIR)

if __name__ == "__main__":
    train()
