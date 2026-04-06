import os
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
import numpy as np
from matplotlib.ticker import FuncFormatter
from typing import List

from models.onsetsandvelocities.ov import OnsetsAndVelocities
from models.onsetsandvelocities.inference import strided_inference, OnsetVelocityNmsDecoder
from preprocessing.constants import (SEQUENCE_LENGTH, DATA_PATH, N_KEYS, N_MELS,
                                     SAMPLE_RATE, HOP_LENGTH, MEL_FMIN, MEL_FMAX)
from preprocessing.dataset import MAESTRO
from preprocessing.mel import MelSpectrogram

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONV1X1_HEAD: List[int] = (200, 200)
BATCH_NORM = 0
LEAKY_RELU_SLOPE: float = 0.1
DROPOUT = 0
IN_CHANS = 2

INFERENCE_CHUNK_SIZE: float = 400
INFERENCE_CHUNK_OVERLAP: float = 11
INFERENCE_THRESHOLD: float = 0.74
FIGSIZE: List[float] = (20, 20)
MEL_CMAP: str = "bone_r"   # cividis
ROLL_CMAP: str = "bone_r"  # binary
TN_RGB: List[int] = (255, 255, 255)
TP_RGB: List[int] = (0, 0, 0)
FP_RGB: List[int] = (179, 179, 255)
FN_RGB: List[int] = (255, 51, 153)
TITLE_SIZE: int = 20
LABEL_SIZE: int = 16
TICK_SIZE: int = 25

SECS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE
CHUNK_SIZE = round(INFERENCE_CHUNK_SIZE / SECS_PER_FRAME)
CHUNK_OVERLAP = round(INFERENCE_CHUNK_OVERLAP / SECS_PER_FRAME)

def qualitative_plot(gt_mel, gt_roll, pred_ons, pred_vel, mel_cmap="binary",
                     roll_cmap="binary", figsize=(10, 40), threshold=0.75,
                     tn_rgb=(255, 255, 255), tp_rgb=(0, 0, 0),
                     fp_rgb=(179, 179, 255), fn_rgb=(255, 51, 153),
                     secs_per_frame=0.024,
                     title_size=38, label_size=32, tick_size=25,
                     ev_title="Ground Truth vs. Thresholded Onset Predictions",
                     min_idx=None, max_idx=None, invert_yaxis=True):
    """
    :param gt_mel: Log-mel spectrogram of shape ``(f, t)``
    :param gt_roll: Ground truth boolean piano roll of shape ``(k, t)``
    :pred_ons: Predicted onsets of shape ``(k, t)`` between 0 and 1
    :pred_vel: Predicted velocities of shape ``(k, t)`` between 0 and 1
    :returns: figure and axes.
    """
    if max_idx is not None:
        gt_mel = gt_mel[:, :max_idx]
        gt_roll = gt_roll[:, :max_idx]
        pred_ons = pred_ons[:, :max_idx]
        pred_vel = pred_vel[:, :max_idx]
    if min_idx is not None:
        gt_mel = gt_mel[:, min_idx:]
        gt_roll = gt_roll[:, min_idx:]
        pred_ons = pred_ons[:, min_idx:]
        pred_vel = pred_vel[:, min_idx:]
    #
    fig, (mel_ax, v_ax, o_ax, eval_ax) = plt.subplots(
        nrows=4, figsize=figsize, sharex=True)
    #
    mel_ax.imshow(gt_mel, cmap=mel_cmap, aspect="auto")
    v_ax.imshow(pred_vel, cmap=roll_cmap, aspect="auto")
    o_ax.imshow(pred_ons, cmap=roll_cmap, aspect="auto")
    #
    pred_mask = (pred_ons >= threshold)
    tp_mask = (gt_roll & pred_mask)
    fp_mask = (~gt_roll & pred_mask)
    fn_mask = (gt_roll & ~pred_mask)
    #
    eval_arr = np.zeros(gt_roll.shape + (3,), dtype=np.uint8)
    eval_arr[:] = tn_rgb
    eval_arr[tp_mask.nonzero()] = tp_rgb
    eval_arr[fp_mask.nonzero()] = fp_rgb
    eval_arr[fn_mask.nonzero()] = fn_rgb
    eval_ax.imshow(eval_arr, aspect="auto")
    # appearance
    mel_ax.set_ylabel("Frequency", fontsize=label_size)
    v_ax.set_ylabel("Key", fontsize=label_size)
    o_ax.set_ylabel("Key", fontsize=label_size)
    eval_ax.set_ylabel("Key", fontsize=label_size)
    eval_ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: f"{x * secs_per_frame:g}"))
    eval_ax.set_xlabel("Time (s)", fontsize=label_size)
    #
    mel_ax.tick_params(labelsize=tick_size)
    v_ax.tick_params(labelsize=tick_size)
    o_ax.tick_params(labelsize=tick_size)
    eval_ax.tick_params(labelsize=tick_size)
    #
    axtitle_pad = 20
    mel_ax.set_title("Input Spectrogram", fontsize=title_size,
                     pad=axtitle_pad)
    v_ax.set_title("Predicted Onset Velocities", fontsize=title_size,
                   pad=axtitle_pad)
    o_ax.set_title("Predicted Onset Probabilities", fontsize=title_size,
                   pad=axtitle_pad)
    eval_ax.set_title(ev_title,
                      fontsize=title_size, pad=axtitle_pad)
    #
    if invert_yaxis:
        mel_ax.invert_yaxis()
        v_ax.invert_yaxis()
        o_ax.invert_yaxis()
        eval_ax.invert_yaxis()
    #
    return fig, (mel_ax, o_ax, v_ax, eval_ax)

def make_triple_onsets(onsets):
    """
    :param onsets: boolean array of shape ``(k, t)``
    :returns: boolean array of same shape, but every true entry at time
      ``t``is also extended to ``t+1, t+2``.
    """
    result = onsets.copy()
    result[:, 1:] |= result[:, :-1]
    result[:, 1:] |= result[:, :-1]
    return result

def load_model(model, path, eval_phase=True, strict=True, to_cpu=False):
    """
    """
    state_dict = torch.load(path, map_location="cpu" if to_cpu else None)
    model.load_state_dict(state_dict, strict=strict)
    if eval_phase:
        model.eval()
    else:
        model.train()

def inference(model_path):
    test_dataset = MAESTRO(DATA_PATH, groups=['test'], sequence_length=SEQUENCE_LENGTH)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    model = OnsetsAndVelocities(
        in_chans=IN_CHANS,
        in_height=N_MELS,
        out_height=N_KEYS,
        conv1x1head=CONV1X1_HEAD,
        bn_momentum=BATCH_NORM,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_drop_p=DROPOUT
    ).to(DEVICE)

    load_model(model, model_path, eval_phase=True, to_cpu=True)

    decoder = OnsetVelocityNmsDecoder(
        num_keys=N_KEYS,
        nms_pool_ksize=3,
        gauss_conv_stddev=1,
        gauss_conv_ksize=11,
        vel_pad_left=1,
        vel_pad_right=1
    ).to(DEVICE)

    ##############
    # INFERENCE
    ##############
    def model_inference(x):
        """
        Convenience wrapper around the DNN to ensure output and input sequences
        have same length.
        """
        with torch.no_grad():
            probs, vels = model(x, trainable_onsets=False)
            probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
            vels = F.pad(torch.sigmoid(vels), (1, 0))
            # df = decoder(probs, vels, INFERENCE_THRESHOLD)
        # probs = torch.zeros_like(probs) # Reinitialize probs to zeros on the same device
        # probs[0][df["key"], df["t_idx"]] = torch.from_numpy(df["vel"].to_numpy()).to(probs.device)
        # print(probs[0])
        # print(df)
        return probs, vels
        # return probs[0], df
    
    idx = random.randint(0, len(test_dataset) - 1)
    sample = test_dataset[idx]
    audio = sample['audio'].unsqueeze(0).to(DEVICE)

    mel_extractor = MelSpectrogram().to(DEVICE)
    mel = mel_extractor(audio)
    onsets = sample['onset'].cpu().numpy().astype(bool).T
    triple_onsets = make_triple_onsets(onsets)

    with torch.no_grad():
        onset_pred, vel_pred = strided_inference(model_inference, mel, CHUNK_SIZE, CHUNK_OVERLAP)

        # onset_pred = onset_pred.to(DEVICE)
        # vel_pred = vel_pred.to(DEVICE)

        # df = decoder(onset_pred, vel_pred, INFERENCE_THRESHOLD)

        onset_pred = onset_pred.cpu().numpy().squeeze()
        vel_pred = vel_pred.cpu().numpy().squeeze()
       
    #     # print(f"Decoded {len(df)} onsets:\n", df)

    # onset_pred *= 0
    # onset_pred[0][df["key"], df["t_idx"]] = torch.from_numpy(df["vel"].to_numpy()).to(onset_pred.device)

    print(onset_pred)
    # print(df)
    # print(onset_pred[0])
    # print(len(onset_pred[0]))

    mel_for_plot = mel.cpu().numpy().squeeze()

    def qplot_ranged(min_idx=None, max_idx=None):
        """
        Closure to inspect ranges flexibly via one-liners like::
            qplot_ranged(0, 1000)[0].show()

        Note that the bottom plot has been adjusted to show only the GT.
        """
        fig, axes = qualitative_plot(
            mel_for_plot, triple_onsets,
            onset_pred, vel_pred,
            mel_cmap=MEL_CMAP,
            roll_cmap=ROLL_CMAP,
            figsize=(20,20),
            threshold=0.74,
            tn_rgb=(255, 255, 255), tp_rgb=(0, 0, 0),
            fp_rgb=(255, 255, 255), fn_rgb=(0, 0, 0),
            secs_per_frame=SECS_PER_FRAME,
            title_size=TITLE_SIZE,
            label_size=LABEL_SIZE,
            tick_size=TICK_SIZE,
            ev_title="Onset Ground Truth",
            min_idx=min_idx, max_idx=max_idx)
        return fig, axes

    fig, _ = qplot_ranged()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example Usage
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    MODEL_PATH = os.path.join(model_dir, 'OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.pt')
    
    # 1. Run inference on random files from the dataset
    print("--- Testing on Dataset ---")
    inference(MODEL_PATH)
    
    # 2. Run inference on a specific file (Uncomment and replace with your actual path)
    # print("--- Testing on Custom Audio File ---")
    # AUDIO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'audio', 'example.wav')
    # infer_audio(MODEL_PATH, AUDIO_FILE_PATH)
