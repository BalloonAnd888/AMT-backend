#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import mir_eval
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.constants import *
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram
from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.decoder import OnsetVelocityNmsDecoder
from models.onsetsandvelocities.visualize import visualize_prediction

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD = (200, 200)
BATCH_NORM = 0.95
LEAKY_RELU_SLOPE = 0.1
DROPOUT = 0.15
IN_CHANS = 2

# Loss Constants
ONSET_POSITIVES_WEIGHT = 8.0
VEL_LOSS_LAMBDA = 10.0

MIN_MIDI = 21

class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        return result

def evaluate(model_path, data_path, batch_size):
    print(f"Loading model from: {model_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {DEVICE}")

    # 1. Model Initialization
    # N_MELS and N_KEYS are expected to be in preprocessing.constants
    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()

    # 2. Data Loading
    # Using 'test' group for evaluation
    # SEQUENCE_LENGTH is expected to be in preprocessing.constants
    test_dataset = MAESTRO(path=data_path, groups=['test'], sequence_length=SEQUENCE_LENGTH)
        
    print(f"test_dataset length: {len(test_dataset)}\n")
    if len(test_dataset) == 0:
        print("ERROR: The dataset is empty. Please check your data path and file extensions.")
        exit()

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Batches: {len(test_dataloader)}")

    mel_extractor = MelSpectrogram().to(DEVICE)

    # Loss functions for reporting
    ons_pos_weights = torch.FloatTensor([ONSET_POSITIVES_WEIGHT]).to(DEVICE)
    ons_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # Initialize Decoder
    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1.0,  # Optional: smooths probabilities before NMS
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

    # Metrics
    metrics = {
        "loss": 0.0,
        "ons_loss": 0.0,
        "vel_loss": 0.0,
        "onset_precision": 0.0,
        "onset_recall": 0.0,
        "onset_f1": 0.0,
        "velocity_precision": 0.0,
        "velocity_recall": 0.0,
        "velocity_f1": 0.0
    }

    with torch.inference_mode():
        progress_bar = tqdm(test_dataloader, desc="Testing")
        for batch in progress_bar:
            audio = batch['audio'].to(DEVICE)
            # Targets: (Batch, Time, Keys) -> (Batch, Keys, Time-1)
            onset = batch['onset'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]
            velocity = batch['velocity'].to(DEVICE).float().permute(0, 2, 1)[:, :, 1:]

            mel = mel_extractor(audio)

            # Forward Pass
            pred_onset_stack, pred_vels = model(mel, trainable_onsets=False)

            # Calculate loss (for reference)
            batch_ons_loss = sum(ons_loss_fn(p, (onset > 0).float()) for p in pred_onset_stack) / len(pred_onset_stack)
            
            vel_mask = (onset > 0).float()
            vel_target = velocity / 127.0
            batch_vel_loss = vel_loss_fn(pred_vels, vel_target, mask=vel_mask)
            
            loss = batch_ons_loss + (batch_vel_loss * VEL_LOSS_LAMBDA)

            # --- DECODER USAGE ---
            # 1. Apply Sigmoid to get probabilities [0, 1]
            pred_probs = torch.sigmoid(pred_onset_stack[-1])
            pred_vels_probs = torch.sigmoid(pred_vels)
            # 2. Decode to DataFrame (columns: batch_idx, key, t_idx, prob, vel)
            decoded_df = decoder(pred_probs, pred_vels_probs, pthresh=0.5)

            # --- MIR_EVAL METRICS ---
            batch_metrics = {k: [] for k in ["onset_p", "onset_r", "onset_f1", "vel_p", "vel_r", "vel_f1"]}
            
            for b in range(audio.shape[0]):
                # 1. Prepare Ground Truth
                # onset[b] is (Keys, T)
                ref_keys, ref_times = torch.where(onset[b] > 0)
                ref_keys = ref_keys.cpu().numpy()
                ref_times = ref_times.cpu().numpy()
                
                ref_pitches = ref_keys + MIN_MIDI
                ref_onsets = ref_times * HOP_LENGTH / SAMPLE_RATE
                # Create intervals [onset, onset + 0.05] as we don't have offsets
                ref_intervals = np.column_stack((ref_onsets, ref_onsets + 0.05))
                # Velocity: normalize 0-127 to 0-1
                ref_vels = velocity[b, ref_keys, ref_times].cpu().numpy() / 127.0

                # 2. Prepare Predictions
                b_df = decoded_df[decoded_df["batch_idx"] == b]
                est_pitches = b_df["key"].values + MIN_MIDI
                est_onsets = b_df["t_idx"].values * HOP_LENGTH / SAMPLE_RATE
                est_intervals = np.column_stack((est_onsets, est_onsets + 0.05))
                est_vels = b_df["vel"].values # Already 0-1 from sigmoid

                # 3. Calculate Onset Metrics
                # offset_ratio=None ignores offsets
                scores = mir_eval.transcription.precision_recall_f1_overlap(
                    ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None
                )
                o_p, o_r, o_f1 = scores[:3]
                
                batch_metrics["onset_p"].append(o_p)
                batch_metrics["onset_r"].append(o_r)
                batch_metrics["onset_f1"].append(o_f1)

                # 4. Calculate Velocity Metrics
                # Find matches based on onset and pitch
                matches = mir_eval.transcription.match_notes(
                    ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None
                )
                
                # Filter matches by velocity tolerance (0.1)
                vel_tp = 0
                for ref_idx, est_idx in matches:
                    if abs(ref_vels[ref_idx] - est_vels[est_idx]) < 0.1:
                        vel_tp += 1
                
                v_p = vel_tp / len(est_pitches) if len(est_pitches) > 0 else 0.0
                v_r = vel_tp / len(ref_pitches) if len(ref_pitches) > 0 else 0.0
                v_f1 = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) > 0 else 0.0

                batch_metrics["vel_p"].append(v_p)
                batch_metrics["vel_r"].append(v_r)
                batch_metrics["vel_f1"].append(v_f1)

            metrics["loss"] += loss.item()
            metrics["ons_loss"] += batch_ons_loss.item()
            metrics["vel_loss"] += batch_vel_loss.item()
            metrics["onset_precision"] += np.mean(batch_metrics["onset_p"])
            metrics["onset_recall"] += np.mean(batch_metrics["onset_r"])
            metrics["onset_f1"] += np.mean(batch_metrics["onset_f1"])
            metrics["velocity_precision"] += np.mean(batch_metrics["vel_p"])
            metrics["velocity_recall"] += np.mean(batch_metrics["vel_r"])
            metrics["velocity_f1"] += np.mean(batch_metrics["vel_f1"])

    # Average metrics
    for k in metrics:
        metrics[k] /= len(test_dataloader)

    print("\n" + "="*40)
    print("TEST RESULTS")
    print("="*40)
    print(f"Total Loss:       {metrics['loss']:.5f}")
    print(f"Onset Loss:       {metrics['ons_loss']:.5f}")
    print(f"Velocity Loss:    {metrics['vel_loss']:.5f}")
    print("-" * 20)
    print(f"Onset Precision:  {metrics['onset_precision']:.4f}")
    print(f"Onset Recall:     {metrics['onset_recall']:.4f}")
    print(f"Onset F1 Score:   {metrics['onset_f1']:.4f}")
    print("-" * 20)
    print(f"Velocity Precision: {metrics['velocity_precision']:.4f}")
    print(f"Velocity Recall:    {metrics['velocity_recall']:.4f}")
    print(f"Velocity F1 Score:  {metrics['velocity_f1']:.4f}")
    print("="*40)

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(graph_dir, f"prediction_vis_{timestamp}.png")
    # visualize_prediction(model, test_dataset, device=DEVICE, save_path=save_path)

if __name__ == "__main__":
    # Default to the models directory relative to this script
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    evaluate(MODEL_PATH, DATA_PATH, batch_size=8)
    