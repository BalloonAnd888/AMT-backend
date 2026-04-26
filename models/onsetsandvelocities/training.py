#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Training script for OnsetsAndVelocities piano transcription model.

Differences from the original repo (iamusica_training):
- Uses on-the-fly mel extraction via MelSpectrogram instead of precomputed HDF5 mels
- Uses the MAESTRO dataset class from preprocessing.dataset instead of MelMaestro
- MelSpectrogram matches the training pipeline: torchaudio MelSpectrogram + AmplitudeToDB

Structured in 3 parts:
1. Configuration and global parameters
2. Instantiation (dataloader, model, decoder, optimizer, loss)
3. Training loop with periodic cross-validation
"""

from datetime import datetime
import os
import random
from dataclasses import dataclass, field
from typing import Optional, List

import functools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset # 
from tqdm import tqdm

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.inference import strided_inference, OnsetVelocityNmsDecoder
from preprocessing.constants import DATA_PATH, N_KEYS, HOP_LENGTH, SAMPLE_RATE, N_MELS, MEL_FMAX, MEL_FMIN, WINDOW_LENGTH, SEQUENCE_LENGTH, MAPS_DATA_PATH, GIANTMIDI_DATA_PATH
from preprocessing.dataset import GIANTMIDI, MAESTRO, MAPS
from preprocessing.mel import MelSpectrogram

# ##############################################################################
# # CONFIGURATION
# ##############################################################################
@dataclass
class TrainConfig:
    """
    All hyperparameters and paths in one place.

    :cvar DATA_PATH: Root directory of the MAESTRO v3 dataset.
    :cvar OUTPUT_DIR: Where to store model snapshots and logs.
    :cvar SNAPSHOT_INPATH: Optional path to a pretrained model to resume from.

    :cvar SAMPLE_RATE: Audio sample rate (must match training data).
    :cvar WIN_SIZE: STFT window size.
    :cvar HOP_SIZE: STFT hop size. Determines time resolution.
    :cvar N_MELS: Number of mel filterbank bins.
    :cvar MEL_FMIN: Lowest mel bin frequency in Hz.
    :cvar MEL_FMAX: Highest mel bin frequency in Hz.

    :cvar TRAIN_BS: Training batch size. Reduce if OOM.
    :cvar TRAIN_BATCH_SECS: Duration of each training chunk in seconds.
    :cvar DATALOADER_WORKERS: Number of DataLoader worker processes.

    :cvar CONV1X1: Hidden layer sizes for the conv1x1 head MLP.
    :cvar BATCH_NORM: Momentum for batch/spectral normalization layers.
    :cvar DROPOUT: Dropout probability.
    :cvar LEAKY_RELU_SLOPE: Negative slope for leaky ReLU.

    :cvar LR_MAX: Peak learning rate.
    :cvar LR_WARMUP: Fraction of first cycle used for warmup.
    :cvar LR_PERIOD: Steps per cosine annealing cycle.
    :cvar LR_DECAY: Multiplier applied to LR limits each cycle.
    :cvar MOMENTUM: Adam beta1.
    :cvar WEIGHT_DECAY: L2 regularization.

    :cvar ONSET_POSITIVES_WEIGHT: Upweight positive onset examples in BCE loss.
    :cvar VEL_LOSS_LAMBDA: Weight of velocity loss relative to onset loss.
    :cvar TRAINABLE_ONSETS: If False, only train the velocity stage.

    :cvar DECODER_GAUSS_STD: Std dev (frames) for NMS Gaussian smoothing.
    :cvar DECODER_GAUSS_KSIZE: Kernel size for NMS Gaussian smoothing.

    :cvar XV_EVERY: Cross-validate every this many global steps.
    :cvar XV_CHUNK_SIZE: Chunk size in seconds for strided XV inference.
    :cvar XV_CHUNK_OVERLAP: Overlap in seconds for strided XV inference.
    :cvar XV_THRESHOLDS: List of onset thresholds to evaluate at XV time.
    :cvar XV_TOLERANCE_SECS: Onset timing tolerance for correct detection.
    :cvar XV_TOLERANCE_VEL: Velocity tolerance for correct detection (0-1).
    """
    # I/O
    DATA_PATH: str = DATA_PATH
    MAPS_DATA_PATH: str = MAPS_DATA_PATH
    GIANTMIDI_DATA_PATH: str = GIANTMIDI_DATA_PATH
    OUTPUT_DIR: str = "out"
    SNAPSHOT_INPATH: Optional[str] = None
    RANDOM_SEED: Optional[int] = None

    # Audio / mel
    SAMPLE_RATE: int = SAMPLE_RATE
    WIN_SIZE: int = WINDOW_LENGTH
    HOP_SIZE: int = HOP_LENGTH
    N_MELS: int = N_MELS
    MEL_FMIN: int = MEL_FMIN
    MEL_FMAX: int = MEL_FMAX

    # Data loader
    TRAIN_BS: int = 4        # lower than original (40) since mels computed on-the-fly
    TRAIN_BATCH_SECS: float = 5.0
    DATALOADER_WORKERS: int = 4

    # Model architecture
    CONV1X1: List[int] = field(default_factory=lambda: [200, 200])
    BATCH_NORM: float = 0.95
    DROPOUT: float = 0.15
    LEAKY_RELU_SLOPE: float = 0.1

    # Optimizer
    LR_MAX: float = 0.008
    LR_WARMUP: float = 0.5
    LR_PERIOD: int = 1000
    LR_DECAY: float = 0.975
    LR_SLOWDOWN: float = 1.0
    MOMENTUM: float = 0.95
    WEIGHT_DECAY: float = 0.0003

    # Loss
    ONSET_POSITIVES_WEIGHT: float = 8.0
    VEL_LOSS_LAMBDA: float = 10.0
    TRAINABLE_ONSETS: bool = True

    # Decoder
    DECODER_GAUSS_STD: float = 1.0
    DECODER_GAUSS_KSIZE: int = 11

    # Training loop
    NUM_EPOCHS: int = 10
    TRAIN_LOG_EVERY: int = 10
    XV_EVERY: int = 1000
    XV_CHUNK_SIZE: float = 600.0
    XV_CHUNK_OVERLAP: float = 2.5
    XV_THRESHOLDS: List[float] = field(default_factory=lambda: [0.7, 0.725, 0.75, 0.775, 0.8])
    XV_TOLERANCE_SECS: float = 0.05
    XV_TOLERANCE_VEL: float = 0.1

# ##############################################################################
# # LOSS HELPERS
# ##############################################################################
class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """BCE loss with an optional per-element mask, averaged over all elements."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            eltwise_loss = eltwise_loss * mask
        return eltwise_loss.mean()


# ##############################################################################
# # OPTIMIZER (cosine annealing with warm restarts + weight decay)
# ##############################################################################
class AdamWR(torch.optim.AdamW):
    """
    AdamW with cosine annealing warm restarts (SGDR-style).
    Mirrors the AdamWR used in the original training repo.
    """
    def __init__(self, params, lr_max=0.01, lr_period=1000,
                 lr_decay=1.0, lr_slowdown=1.0, lr_warmup=0.0,
                 cycle_end_hook_fn=None, **adamw_kwargs):
        super().__init__(params, lr=lr_max, **adamw_kwargs)
        self.lr_max = lr_max
        self.lr_min = lr_max * 1e-4
        self.lr_period = lr_period
        self.lr_decay = lr_decay
        self.lr_slowdown = lr_slowdown
        self.lr_warmup = lr_warmup  # fraction of cycle used for warmup
        self.cycle_end_hook_fn = cycle_end_hook_fn
        self._step = 0
        self._cycle = 0
        self._cycle_step = 0
        self._cycle_len = lr_period

    def get_lr(self):
        return self.param_groups[0]['lr']

    def step(self, closure=None):
        # compute LR for this step
        cs = self._cycle_step
        cl = self._cycle_len
        warmup_steps = int(self.lr_warmup * cl)

        if cs < warmup_steps:
            # linear warmup
            lr = self.lr_min + (self.lr_max - self.lr_min) * (cs / max(1, warmup_steps))
        else:
            # cosine decay
            progress = (cs - warmup_steps) / max(1, cl - warmup_steps)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(np.pi * progress))

        for pg in self.param_groups:
            pg['lr'] = lr

        super().step(closure)

        self._step += 1
        self._cycle_step += 1

        if self._cycle_step >= self._cycle_len:
            # end of cycle
            self._cycle += 1
            self._cycle_step = 0
            self._cycle_len = round(self._cycle_len * self.lr_slowdown)
            self.lr_max *= self.lr_decay
            self.lr_min = self.lr_max * 1e-4
            if self.cycle_end_hook_fn is not None:
                self.cycle_end_hook_fn()


# ##############################################################################
# # MODEL SAVE / LOAD
# ##############################################################################
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model_weights(model, path, eval_phase=True):
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval() if eval_phase else model.train()
    print(f"Loaded weights from {path}")


# ##############################################################################
# # COLLATE: audio → mel on-the-fly in the DataLoader worker
# ##############################################################################
def _collate_fn_impl(batch):
    """
    Helper function for collating a batch.
    1. Stacks raw audio samples into a batch
    2. Peak-normalizes the batch
    3. Returns (audio, onset_rolls, velocity_rolls)
    """
    # batch is a list of dicts from MAESTRO.__getitem__
    audio = torch.stack([b['audio'] for b in batch])          # (B, samples)
    onsets = torch.stack([b['onset'] for b in batch])         # (B, t, keys)
    velocities = torch.stack([b['velocity'] for b in batch])  # (B, t, keys)

    # Peak-normalize each sample independently (matches training preprocessing)
    maxvals = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    audio = audio / maxvals

    return audio, onsets, velocities

def make_collate_fn():
    """
    Returns a collate function that:
    1. Stacks raw audio samples into a batch
    2. Returns (audio, onset_rolls, velocity_rolls)
    """
    return _collate_fn_impl


# ##############################################################################
# # XV HELPERS
# ##############################################################################
def eval_note_events_simple(gt_onsets_sec, gt_keys, pred_onsets_frames,
                             pred_keys, secs_per_frame,
                             tol_secs=0.05, key_shift=0):
    """
    Simple precision/recall/F1 for onset detection.
    Matches a predicted onset to a GT onset if same key and |dt| <= tol_secs.
    Each GT and pred onset can be matched at most once.
    """
    pred_secs = np.array(pred_onsets_frames, dtype=float) * secs_per_frame
    pred_keys_arr = np.array(pred_keys, dtype=int) + key_shift

    matched_gt = set()
    matched_pred = set()

    gt_pairs = list(zip(gt_onsets_sec, gt_keys))
    pred_pairs = list(zip(pred_secs, pred_keys_arr))

    for pi, (pt, pk) in enumerate(pred_pairs):
        for gi, (gt, gk) in enumerate(gt_pairs):
            if gi in matched_gt:
                continue
            if pk == gk and abs(pt - gt) <= tol_secs:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    prec = tp / len(pred_pairs) if pred_pairs else 0.0
    rec = tp / len(gt_pairs) if gt_pairs else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ##############################################################################
# # MAIN
# ##############################################################################
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    CONF = TrainConfig()

    # Allow CLI overrides: python train.py TRAIN_BS=8 LR_MAX=0.004
    import sys
    for arg in sys.argv[1:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            if hasattr(CONF, k):
                attr = getattr(CONF, k)
                try:
                    setattr(CONF, k, type(attr)(v))
                except (ValueError, TypeError):
                    print(f"Warning: could not parse CLI arg {arg}")

    if CONF.RANDOM_SEED is None:
        CONF.RANDOM_SEED = random.randint(0, int(1e7))
    random.seed(CONF.RANDOM_SEED)
    np.random.seed(CONF.RANDOM_SEED)
    torch.manual_seed(CONF.RANDOM_SEED)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    SECS_PER_FRAME = CONF.HOP_SIZE / CONF.SAMPLE_RATE
    # SEQUENCE_LENGTH = round(CONF.TRAIN_BATCH_SECS * CONF.SAMPLE_RATE)
    XV_CHUNK_SIZE = round(CONF.XV_CHUNK_SIZE / SECS_PER_FRAME)
    XV_CHUNK_OVERLAP = round(CONF.XV_CHUNK_OVERLAP / SECS_PER_FRAME)
    MIN_MIDI = 21  # A0

    # Output dirs
    model_snapshot_dir = os.path.join(CONF.OUTPUT_DIR, "model_snapshots")
    log_dir = os.path.join(CONF.OUTPUT_DIR, "logs")
    os.makedirs(model_snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train_log.txt")

    timestamp = datetime.now().strftime('%y%m%d-%H%M%S')

    def log(tag, msg):
        line = f"[{tag}] {msg}"
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log("CONFIG", vars(CONF))

    dataset = "giantmidi" # "maestro", "maps", "giantmidi"

    # ------------------------------------------------------------------
    # Mel extractor (GPU)
    # ------------------------------------------------------------------
    mel_extractor = MelSpectrogram(
        sample_rate=CONF.SAMPLE_RATE,
        window_length=CONF.WIN_SIZE,
        hop_length=CONF.HOP_SIZE,
        n_mels=CONF.N_MELS,
        fmin=CONF.MEL_FMIN,
        fmax=CONF.MEL_FMAX
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    if dataset == "maestro":
        print("Loading training dataset...")
        train_dataset = MAESTRO(CONF.DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH, device='cpu')
        print("Loading validation dataset...")
        xv_dataset = MAESTRO(CONF.DATA_PATH, groups=['validation'], sequence_length=None, device='cpu')   # full files for XV
    elif dataset == "maps":
        print("Loading training dataset...")
        train_dataset = MAPS(path=CONF.MAPS_DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH, device='cpu')
        print("Loading validation dataset...")
        xv_dataset = MAPS(path=CONF.MAPS_DATA_PATH, groups=['validation'], sequence_length=None, device='cpu')
    elif dataset == "giantmidi":
        print("Loading training dataset...")
        train_dataset = GIANTMIDI(path=CONF.GIANTMIDI_DATA_PATH, groups=['train'], sequence_length=SEQUENCE_LENGTH, device='cpu')
        print("Loading validation dataset...")
        xv_dataset = GIANTMIDI(path=CONF.GIANTMIDI_DATA_PATH, groups=['validation'], sequence_length=None, device='cpu')

    train_size = len(train_dataset) // 10 #
    train_dataset = Subset(train_dataset, range(train_size)) #

    val_size = len(xv_dataset) // 10
    xv_dataset = Subset(xv_dataset, range(val_size))

    collate_fn = make_collate_fn()

    train_dl = DataLoader(
        train_dataset,
        batch_size=CONF.TRAIN_BS,
        shuffle=True,
        num_workers=CONF.DATALOADER_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=False)

    batches_per_epoch = len(train_dl)
    log("DATASET", f"Train: {len(train_dataset)} samples, "
                   f"{batches_per_epoch} batches/epoch")
    log("DATASET", f"XV: {len(xv_dataset)} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = OnsetsAndVelocities(
        in_chans=2,           # mel + time_derivative(mel), built inside model
        in_height=CONF.N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONF.CONV1X1,
        bn_momentum=CONF.BATCH_NORM,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE,
        dropout_drop_p=CONF.DROPOUT
    ).to(DEVICE)

    if CONF.SNAPSHOT_INPATH is not None:
        load_model_weights(model, CONF.SNAPSHOT_INPATH, eval_phase=False)

    # ------------------------------------------------------------------
    # Decoder (stays on CPU — used only during XV)
    # ------------------------------------------------------------------
    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=CONF.DECODER_GAUSS_STD,
        gauss_conv_ksize=CONF.DECODER_GAUSS_KSIZE,
        vel_pad_left=1,
        vel_pad_right=1
    )  # CPU

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    ons_pos_weights = torch.FloatTensor([CONF.ONSET_POSITIVES_WEIGHT]).to(DEVICE)
    ons_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=ons_pos_weights)
    vel_loss_fn = MaskedBCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    trainable_params = (model.parameters() if CONF.TRAINABLE_ONSETS
                        else model.velocity_stage.parameters())

    def on_cycle_end():
        """Called at end of each LR cycle — save a model snapshot."""
        step = global_step
        path = os.path.join(model_snapshot_dir,
                            f"OnsetsAndVelocities_step={step}.pt")
        save_model(model, path)

    optimizer = AdamWR(
        trainable_params,
        lr_max=CONF.LR_MAX,
        lr_period=CONF.LR_PERIOD,
        lr_decay=CONF.LR_DECAY,
        lr_slowdown=CONF.LR_SLOWDOWN,
        lr_warmup=CONF.LR_WARMUP,
        cycle_end_hook_fn=on_cycle_end,
        weight_decay=CONF.WEIGHT_DECAY,
        betas=(CONF.MOMENTUM, 0.999),
        eps=1e-8,
        amsgrad=False)

    # ------------------------------------------------------------------
    # XV inference helper
    # ------------------------------------------------------------------
    def model_inference_xv(x):
        """Wrapper for strided_inference during XV. x: (1, n_mels, t)"""
        probs, vels = model(x, trainable_onsets=False)
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        return probs, vels

    def xv_single_file(sample, thresholds):
        """
        Run inference on one full-length XV file and evaluate at all thresholds.
        Returns list of (prec, rec, f1) tuples, one per threshold.
        """
        audio = sample['audio'].to(DEVICE)
        audio = audio / (audio.abs().max() + 1e-8)

        with torch.no_grad():
            mel = mel_extractor(audio)
            mel = mel.reshape(1, CONF.N_MELS, -1)
            onset_pred, vel_pred = strided_inference(
                model_inference_xv, mel, XV_CHUNK_SIZE, XV_CHUNK_OVERLAP)

        # Build ground truth onset list from label tensor
        dataset_secs_per_frame = HOP_LENGTH / CONF.SAMPLE_RATE
        label = sample['label']  # (t, N_KEYS) uint8
        gt_onsets_frames = []
        gt_keys = []
        for key_idx in range(N_KEYS):
            frames = (label[:, key_idx] == 3).nonzero(as_tuple=True)[0]
            for f in frames.tolist():
                gt_onsets_frames.append(f)
                gt_keys.append(key_idx)
        gt_onsets_sec = np.array(gt_onsets_frames) * dataset_secs_per_frame

        results = []
        for t in thresholds:
            df = decoder(onset_pred.cpu(), vel_pred.cpu(), t)
            if len(df) == 0:
                results.append((0.0, 0.0, 0.0))
                continue
            prec, rec, f1 = eval_note_events_simple(
                gt_onsets_sec, np.array(gt_keys),
                df['t_idx'].to_numpy(), df['key'].to_numpy(),
                SECS_PER_FRAME,
                tol_secs=CONF.XV_TOLERANCE_SECS,
                key_shift=0)
            results.append((prec, rec, f1))
        return results

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log("START", f"Training for {CONF.NUM_EPOCHS} epochs, "
                 f"seed={CONF.RANDOM_SEED}, device={DEVICE}")

    global_step = 1

    for epoch in range(1, CONF.NUM_EPOCHS + 1):
        model.train()

        for batch_idx, (audio, onsets_raw, velocities_raw) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{CONF.NUM_EPOCHS}")):

            # --------------------------------------------------------------
            # Cross-validation
            # --------------------------------------------------------------
            if (global_step % CONF.XV_EVERY) == 0:
                model.eval()
                torch.cuda.empty_cache()

                all_results = {t: [] for t in CONF.XV_THRESHOLDS}

                with torch.no_grad():
                    for xv_idx in tqdm(range(len(xv_dataset)),
                                       desc=f"XV step={global_step}"):
                        sample = xv_dataset[xv_idx]
                        try:
                            results = xv_single_file(sample, CONF.XV_THRESHOLDS)
                            for t, (p, r, f) in zip(CONF.XV_THRESHOLDS, results):
                                all_results[t].append(f)
                        except Exception as e:
                            log("XV_ERROR", f"idx={xv_idx}: {e}")

                # Find best threshold
                avg_f1s = {t: np.mean(v) if v else 0.0
                           for t, v in all_results.items()}
                best_t = max(avg_f1s, key=avg_f1s.get)
                best_f1 = avg_f1s[best_t]

                log("XV_SUMMARY", {
                    "epoch": epoch,
                    "global_step": global_step,
                    "f1_by_threshold": {str(t): f"{v:.4f}"
                                        for t, v in avg_f1s.items()},
                    "best_threshold": best_t,
                    "best_f1": f"{best_f1:.4f}"
                })

                # Save snapshot with F1 in the name
                snap_path = os.path.join(
                    model_snapshot_dir,
                    f"OnsetsAndVelocities_step={global_step}"
                    f"_f1={best_f1:.4f}_t={best_t}.pt")
                save_model(model, snap_path)

                torch.cuda.empty_cache()
                model.train()

            # --------------------------------------------------------------
            # Forward + loss
            # --------------------------------------------------------------
            audio = audio.to(DEVICE)
            with torch.no_grad():
                logmels = mel_extractor(audio)         # (B, n_mels, t)

            # onsets_raw: (B, t, N_KEYS) float from dataset (0 or 1)
            # velocities_raw: (B, t, N_KEYS) float in [0, 1]
            # Model output is (B, N_KEYS, t-1), so we align targets accordingly
            onsets_t = onsets_raw.permute(0, 2, 1).to(DEVICE)       # (B, N_KEYS, t)
            velocities_t = velocities_raw.permute(0, 2, 1).to(DEVICE)

            # Drop first time frame to match model output (forward_onsets uses diff)
            onsets_t = onsets_t[:, :, 1:]          # (B, N_KEYS, t-1)
            velocities_t = velocities_t[:, :, 1:]

            optimizer.zero_grad()

            onset_stages, velocities_pred = model(logmels, CONF.TRAINABLE_ONSETS)

            # Interpolate targets to match the model's output time dimension
            target_time_steps = velocities_pred.shape[-1]
            if onsets_t.shape[-1] != target_time_steps:
                onsets_t = F.interpolate(onsets_t.float(), size=target_time_steps, mode='nearest')
                velocities_t = F.interpolate(velocities_t.float(), size=target_time_steps, mode='nearest')

            # Triple onsets (extend each onset by 2 frames) to match training
            double_onsets = onsets_t.clone()
            torch.maximum(onsets_t[..., :-1], onsets_t[..., 1:],
                          out=double_onsets[..., 1:])
            triple_onsets = double_onsets.clone()
            torch.maximum(double_onsets[..., :-1], double_onsets[..., 1:],
                          out=triple_onsets[..., 1:])

            onsets_clip = triple_onsets.clamp(0, 1)      # binary targets
            onsets_norm = triple_onsets * velocities_t   # velocity-weighted targets

            vel_loss = CONF.VEL_LOSS_LAMBDA * vel_loss_fn(
                velocities_pred, onsets_norm, mask=onsets_clip)
            loss = vel_loss

            if CONF.TRAINABLE_ONSETS:
                ons_loss = sum(ons_loss_fn(ons, onsets_clip)
                               for ons in onset_stages) / len(onset_stages)
                loss = loss + ons_loss

            loss.backward()
            optimizer.step()

            # --------------------------------------------------------------
            # Logging
            # --------------------------------------------------------------
            if (global_step % CONF.TRAIN_LOG_EVERY) == 0:
                losses = {"vel_loss": f"{vel_loss.item():.4f}"}
                if CONF.TRAINABLE_ONSETS:
                    losses["ons_loss"] = f"{ons_loss.item():.4f}"
                losses["total"] = f"{loss.item():.4f}"
                log("TRAIN", {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "batches_per_epoch": batches_per_epoch,
                    "losses": losses,
                    "lr": f"{optimizer.get_lr():.6f}"
                })

            global_step += 1

        # End of epoch — save checkpoint
        epoch_path = os.path.join(model_snapshot_dir,
                                  f"OnsetsAndVelocities-{dataset}-{timestamp}-epoch={epoch}.pt")
        save_model(model, epoch_path)
        log("EPOCH_END", f"Epoch {epoch}/{CONF.NUM_EPOCHS} complete")

    log("DONE", "Training complete.")