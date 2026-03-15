#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt
import torch
import librosa.display
from preprocessing.mel import MelSpectrogram
from preprocessing.constants import *

def plot_learning_curves(history: dict[str, list], timestamp: str, save_dir: str):
    """
    Plots learning curves for loss and accuracy.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 15))
    
    # 1. Total Loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # 2. Onset Loss
    plt.subplot(3, 2, 3)
    plt.plot(epochs, history["train_ons_loss"], label="Train")
    plt.plot(epochs, history["val_ons_loss"], label="Validation")
    plt.title("Onset Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 3. Velocity Loss
    plt.subplot(3, 2, 4)
    plt.plot(epochs, history["train_vel_loss"], label="Train")
    plt.plot(epochs, history["val_vel_loss"], label="Validation")
    plt.title("Velocity Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # 4. Onset Accuracy
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history["train_ons_acc"], label="Train")
    plt.plot(epochs, history["val_ons_acc"], label="Validation")
    plt.title("Onset Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # 5. Velocity Accuracy
    plt.subplot(3, 2, 6)
    plt.plot(epochs, history["train_vel_acc"], label="Train")
    plt.plot(epochs, history["val_vel_acc"], label="Validation")
    plt.title("Velocity Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
 
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"learning_curves_{timestamp}.png")
        plt.savefig(save_path)
        print(f"Learning curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

    if save_dir:
        for title, train_key, val_key, ylabel, filename_suffix in [
            ("Total Loss", "train_loss", "val_loss", "Loss", "total_loss"),
            ("Onset Loss", "train_ons_loss", "val_ons_loss", "Loss", "onset_loss"),
            ("Velocity Loss", "train_vel_loss", "val_vel_loss", "Loss", "velocity_loss"),
            ("Onset Accuracy", "train_ons_acc", "val_ons_acc", "Accuracy", "onset_acc"),
            ("Velocity Accuracy", "train_vel_acc", "val_vel_acc", "Accuracy", "velocity_acc"),
        ]:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history[train_key], label="Train")
            plt.plot(epochs, history[val_key], label="Validation")
            plt.title(title)
            plt.xlabel("Epochs")
            plt.ylabel(ylabel)
            plt.legend()
            
            save_path = os.path.join(save_dir, f"{filename_suffix}_{timestamp}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {title} graph to: {save_path}")

def visualize_data(data):
    """
    Visualizes a single sample from the dataset: Mel Spectrogram, Frames, Onsets, and Velocity.
    """
    audio = data['audio']
    frames = data['frame']
    onsets = data['onset']
    velocity = data['velocity']

    # Ensure audio is on the correct device and has batch dimension for MelSpectrogram
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    mel_extractor = MelSpectrogram().to(DEVICE)
    with torch.inference_mode():
        mel = mel_extractor(audio.to(DEVICE))
    
    # Move to CPU and numpy for plotting
    mel = mel.squeeze(0).cpu().numpy()
    frames = frames.cpu().numpy().T
    onsets = onsets.cpu().numpy().T
    velocity = velocity.cpu().numpy().T

    fig, ax = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    if 'path' in data:
        fig.suptitle(f"File: {os.path.basename(data['path'])}", fontsize=16)

    # 1. Mel Spectrogram
    librosa.display.specshow(
        mel, 
        sr=SAMPLE_RATE, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='mel', 
        fmin=MEL_FMIN, 
        fmax=MEL_FMAX, 
        ax=ax[0],
        cmap='magma'
    )
    ax[0].set_title("Mel Spectrogram")

    duration = len(audio.squeeze()) / SAMPLE_RATE
    extent = [0, duration, 0, N_KEYS]

    # 2. Frames
    ax[1].imshow(frames, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
    ax[1].set_title("Frames (Sustain)")
    ax[1].set_ylabel("Piano Key")

    # 3. Onsets
    ax[2].imshow(onsets, origin='lower', aspect='auto', interpolation='nearest', cmap='Greys', extent=extent)
    ax[2].set_title("Onsets")
    ax[2].set_ylabel("Piano Key")

    # 4. Velocity
    im = ax[3].imshow(velocity, origin='lower', aspect='auto', interpolation='nearest', cmap='magma', extent=extent)
    ax[3].set_title("Velocity")
    ax[3].set_ylabel("Piano Key")
    ax[3].set_xlabel("Time (s)")
    plt.colorbar(im, ax=ax[3])

    plt.tight_layout()
    plt.show()
