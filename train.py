#!/usr/bin/env python3
"""
Underwater SPAD LiDAR dToF distance measurement with U-Net + MC Dropout
and baselines (MLP, 1D ResNet, 1D Transformer).

Use --bench to train/evaluate all baselines sequentially.
Use --model to pick a single model.
"""

import argparse
import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import erf
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# -----------------------------------------
# Simulator
# -----------------------------------------
class UnderwaterSPADSimulator:
    def __init__(self, bin_resolution_ps=104.17, lidar_period_bins=1600, laser_pulse_fwhm_ps=200):
        self.bin_res = bin_resolution_ps * 1e-12
        self.num_bins = int(lidar_period_bins)
        self.period_duration = self.num_bins * self.bin_res
        self.c_vacuum = 3e8
        self.refractive_index_water = 1.33
        self.c_water = self.c_vacuum / self.refractive_index_water
        self.sigma_pulse = (laser_pulse_fwhm_ps * 1e-12) / (2 * np.sqrt(2 * np.log(2)))
        self.time_axis = np.linspace(0, self.period_duration, self.num_bins)
        self.dist_axis = 0.5 * self.c_water * self.time_axis

    def compute_bin_probabilities(self, mu_t, sigma_total, signal_photons):
        t_start = self.time_axis
        t_end = self.time_axis + self.bin_res
        denom = sigma_total * np.sqrt(2)
        prob_dist = 0.5 * (erf((t_end - mu_t) / denom) - erf((t_start - mu_t) / denom))
        return prob_dist * signal_photons

    def generate_sample(self, target_bin_index=None, turbidity_level="medium",
                        signal_strength="random", difficulty="any"):
        # difficulty -> sets turbidity, ambient, reflectivity ranges
        if difficulty == "easy":
            turbidity_level = np.random.choice(["low", "medium"], p=[0.7, 0.3])
            ambient_level = np.random.uniform(0.05, 1.0)
            base_reflectivity = np.random.uniform(150, 300)
        elif difficulty == "medium":
            turbidity_level = np.random.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])
            ambient_level = np.random.uniform(0.1, 5.0)
            base_reflectivity = np.random.uniform(80, 220)
        elif difficulty == "hard":
            turbidity_level = np.random.choice(["medium", "high"], p=[0.4, 0.6])
            ambient_level = np.random.uniform(2.0, 15.0)
            base_reflectivity = np.random.uniform(20, 120)
        else:
            ambient_level = np.random.uniform(0.1, 10)
            base_reflectivity = np.random.uniform(50, 200)

        # turbidity parameters
        if turbidity_level == "low":
            attenuation_c, backscatter_amp, backscatter_decay = 1/20, 1, 2.0
        elif turbidity_level == "medium":
            attenuation_c, backscatter_amp, backscatter_decay = 1/5, 2, 4.0
        else:
            attenuation_c, backscatter_amp, backscatter_decay = 1/0.5, 5, 6.0

        # target depth: favor close range (< num_bins/8) but allow occasional farther targets (< num_bins/2)
        if target_bin_index is None:
            near_limit = max(11, int(self.num_bins / 8))
            far_limit = max(near_limit + 1, int(self.num_bins / 2))
            if np.random.rand() < 0.8:
                target_bin_index = np.random.randint(10, near_limit)
            else:
                target_bin_index = np.random.randint(10, far_limit)
        target_time = self.time_axis[target_bin_index]
        target_dist = self.dist_axis[target_bin_index]

        # signal strength
        if signal_strength != "random":
            base_reflectivity = signal_strength
        geom_loss = 1.0 / (max(target_dist, 0.5) ** 2)
        trans_loss = np.exp(-2 * attenuation_c * target_dist)
        avg_signal_photons = base_reflectivity * geom_loss * trans_loss

        # signal component
        signal_profile = self.compute_bin_probabilities(target_time, self.sigma_pulse, avg_signal_photons)

        # backscatter and noise
        backscatter_profile = backscatter_amp * np.exp(-backscatter_decay * self.time_axis * 1e8)
        noise_floor = np.full(self.num_bins, ambient_level)

        # total rate
        total_rate = signal_profile + backscatter_profile + noise_floor
        noisy_histogram = np.random.poisson(total_rate)

        # ground truth label (normalized soft Gaussian)
        label_sigma = 3 * self.bin_res
        gt = self.compute_bin_probabilities(target_time, label_sigma, 1.0)
        if np.sum(gt) > 0:
            gt /= np.sum(gt)

        # normalize input
        max_val = np.max(noisy_histogram)
        norm_input = noisy_histogram / max_val if max_val > 0 else noisy_histogram

        metadata = {
            "target_dist_m": target_dist,
            "signal_photons": avg_signal_photons,
            "total_counts": np.sum(noisy_histogram),
            "turbidity": turbidity_level,
        }
        return norm_input, gt, metadata

# ==========================================
# PART 2: Models
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, p_drop=0.1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 16, p_drop)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(16, 32, p_drop))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64, p_drop))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128, p_drop))
        self.bot   = nn.Sequential(nn.MaxPool1d(2), DoubleConv(128, 256, p_drop))
        self.up1 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128, p_drop)
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64, p_drop)
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32, p_drop)
        self.up4 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(32, 16, p_drop)
        self.outc = nn.Conv1d(16, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        return self.sigmoid(self.outc(x))

class NaiveMLP(nn.Module):
    def __init__(self, L=1600, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, L),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.net(x)
        return y.unsqueeze(1)

class ResBlock1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Conv1d(c, c, 3, padding=1),
            nn.BatchNorm1d(c),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.block(x))

class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, base=64, blocks=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock1D(base) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv1d(base, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return self.head(x)

class Transformer1D(nn.Module):
    def __init__(self, L=1600, d_model=128, nhead=4, num_layers=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.L = L
        self.in_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
    def forward(self, x):
        x = x.transpose(1, 2)          # [B, L, 1]
        x = self.in_proj(x) + self.pos # [B, L, d_model]
        x = self.enc(x)
        x = self.out_proj(x)           # [B, L, 1]
        x = self.sigmoid(x)
        return x.transpose(1, 2)       # [B, 1, L]

# --- Additional baselines ---

class LeNet1D(nn.Module):
    def __init__(self, L=1600):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(4, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        reduced_L = L // 4  # two pools
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * reduced_L, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, L),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.unsqueeze(1)


class AlexNet1D(nn.Module):
    def __init__(self, L=1600):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, L),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.unsqueeze(1)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, dilation=1):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=stride,
                                   padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MobileNet1D(nn.Module):
    def __init__(self, width_mult=1.0, L=1600):
        super().__init__()
        def c(ch):
            return max(8, int(ch * width_mult))

        layers = [nn.Conv1d(1, c(32), 3, padding=1, stride=2, bias=False),
                  nn.BatchNorm1d(c(32)), nn.ReLU(inplace=True)]
        cfg = [ (c(32), c(64), 1), (c(64), c(128), 2), (c(128), c(128), 1),
                (c(128), c(256), 2), (c(256), c(256), 1), (c(256), c(512), 2) ]
        for inp, oup, s in cfg:
            layers.append(DepthwiseSeparableConv1d(inp, oup, k=3, stride=s))
        self.features = nn.Sequential(*layers)
        reduced_L = L // 2
        reduced_L = reduced_L // 2 // 2 // 2  # account for stride-2 layers
        reduced_L = max(reduced_L, 1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c(512), L),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.unsqueeze(1)


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        # remove future leakage for causal conv
        out = out[:, :, :- (out.size(2) - x.size(2))] if out.size(2) != x.size(2) else out
        return self.act(out + self.downsample(x))


class TCN1D(nn.Module):
    def __init__(self, L=1600, channels=(32, 64, 128), k=3, dropout=0.1):
        super().__init__()
        layers = []
        c_prev = 1
        dilation = 1
        for c in channels:
            layers.append(TCNBlock(c_prev, c, k=k, dilation=dilation, dropout=dropout))
            c_prev = c
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv1d(c_prev, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.head(x)
        return x


class PerformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _phi(x):
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, x):  # x: [B, L, d_model]
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.d_head)
        k = self.k_proj(x).view(B, L, self.nhead, self.d_head)
        v = self.v_proj(x).view(B, L, self.nhead, self.d_head)
        q_phi = self._phi(q)
        k_phi = self._phi(k)
        kv = torch.einsum('blhd,blhe->bhde', k_phi, v)  # [B, head, d, d]
        z = 1.0 / (torch.einsum('blhd,bhd->blh', q_phi, k_phi.sum(dim=1)) + 1e-6)
        out = torch.einsum('blhd,bhde,blh->blhe', q_phi, kv, z)
        out = out.reshape(B, L, -1)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class Performer1D(nn.Module):
    def __init__(self, L=1600, d_model=128, nhead=4, num_layers=4, dim_ff=256):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        self.blocks = nn.ModuleList([PerformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)])
        self.out_proj = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.in_proj(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        return x.transpose(1, 2)


class LocalAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, window=64, dropout=0.1):
        super().__init__()
        self.window = window
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, x):  # x: [B, L, d]
        L = x.size(1)
        idx = torch.arange(L, device=x.device)
        dist = idx[None, :] - idx[:, None]
        mask = dist.abs() > self.window
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        return attn_output


class LongformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, window=64, dim_ff=256, dropout=0.1):
        super().__init__()
        self.attn = LocalAttention(d_model, nhead, window, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class Longformer1D(nn.Module):
    def __init__(self, L=1600, d_model=128, nhead=4, num_layers=4, window=64, dim_ff=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, L, d_model))
        self.blocks = nn.ModuleList([
            LongformerBlock(d_model, nhead, window, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.in_proj(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        return x.transpose(1, 2)

# ==========================================
# MC Dropout Helpers
# ==========================================

def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

@torch.no_grad()
def mc_predict(model, x, T=30):
    model.eval()
    enable_mc_dropout(model)
    preds = []
    for _ in range(T):
        preds.append(model(x))
    preds = torch.stack(preds, dim=0)  # [T, B, 1, L]
    mean = preds.mean(dim=0)
    var = preds.var(dim=0, unbiased=False)
    return mean, var

# ==========================================
# Dataset builder with difficulty mix
# ==========================================

def build_dataset(sim, n, difficulty_mix):
    X, Y = [], []
    diffs = list(difficulty_mix.keys())
    probs = list(difficulty_mix.values())
    for _ in range(n):
        diff = np.random.choice(diffs, p=probs)
        x, y, _ = sim.generate_sample(difficulty=diff)
        X.append(x)
        Y.append(y)
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, Y)

# ==========================================
# Training & Evaluation
# ==========================================

def train_one(model, train_loader, val_loader, device, criterion, optimizer, num_epochs,
             is_mc=False, mc_samples=30, checkpoint_path=None):
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    patience = 0
    tolerance = 1e-8
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_tr = running / len(train_loader)
        train_losses.append(avg_tr)

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                if is_mc:
                    mean_pred, _ = mc_predict(model, xb, T=mc_samples)
                    out = mean_pred
                else:
                    out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item()
            avg_val = vloss / len(val_loader)
            val_losses.append(avg_val)
        logger.info("Epoch [%s/%s] Train %.8f | Val %.8f", epoch + 1, num_epochs, avg_tr, avg_val)
        if avg_val < best_val - tolerance:
            best_val = avg_val
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
                logger.info("New best val loss %.8f. Checkpoint saved to %s", best_val, checkpoint_path)
        else:
            patience += 1
            if patience >= 10:
                logger.info("Early stopping triggered after %s epochs without significant val loss improvement.", patience)
                break
    return train_losses, val_losses, best_val, best_state

def evaluate(model, val_loader, device, criterion, is_mc=False, mc_samples=30, bin_to_meters=0.01174):
    model.eval()
    val_losses = []
    distance_errors = []
    snr_improvements = []
    dist_uncertainties = []  # only for MC

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs_dev, targets_dev = inputs.to(device), targets.to(device)
            if is_mc:
                mean_pred, var_pred = mc_predict(model, inputs_dev, T=mc_samples)
                outputs = mean_pred
            else:
                outputs = model(inputs_dev)
            loss = criterion(outputs, targets_dev)
            val_losses.append(loss.item())

            preds_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            inputs_np = inputs.cpu().numpy()

            # distance metrics
            for i in range(len(preds_np)):
                pred_curve = preds_np[i, 0]
                true_curve = targets_np[i, 0]
                bins = np.arange(pred_curve.shape[0])
                mask_pred = pred_curve > (0.1 * pred_curve.max())
                if np.sum(pred_curve[mask_pred]) > 0:
                    pred_centroid = np.sum(bins[mask_pred] * pred_curve[mask_pred]) / np.sum(pred_curve[mask_pred])
                else:
                    pred_centroid = np.argmax(pred_curve)
                true_peak = np.argmax(true_curve)
                error = abs(pred_centroid - true_peak)
                distance_errors.append(error)

                # SNR improvement
                input_noise = np.std(inputs_np[i, 0, :100])
                input_signal = np.max(inputs_np[i, 0])
                output_signal = np.max(pred_curve)
                output_noise = np.std(pred_curve[:100])
                if input_noise > 0 and output_noise > 0:
                    input_snr = input_signal / input_noise
                    output_snr = output_signal / output_noise
                    snr_improvements.append(output_snr / input_snr if input_snr > 0 else 1.0)

                # distance uncertainty if MC
                if is_mc:
                    # recompute centroids per MC sample
                    mean_pred, var_pred = mc_predict(model, inputs_dev[i:i+1], T=mc_samples)
                    mc_preds = mean_pred  # [1,1,L]
                    # For simplicity, centroid on each MC forward pass is not returned by mc_predict
                    # Approximate with normal noise from variance? Instead, re-run full MC:
                    mc_curves = []
                    for _ in range(mc_samples):
                        mc_curves.append(model(inputs_dev[i:i+1]).cpu().numpy()[0,0])
                    mc_curves = np.stack(mc_curves, axis=0)
                    mc_centroids = []
                    for c in mc_curves:
                        m = c > 0.1 * c.max()
                        if np.sum(c[m]) > 0:
                            mc_centroids.append(np.sum(bins[m] * c[m]) / np.sum(c[m]))
                        else:
                            mc_centroids.append(np.argmax(c))
                    dist_uncertainties.append(np.std(mc_centroids))

    avg_val_loss = np.mean(val_losses)
    median_error = np.median(distance_errors)
    mean_error = np.mean(distance_errors)
    std_error = np.std(distance_errors)
    mae = np.mean(distance_errors)
    rmse = np.sqrt(np.mean(np.array(distance_errors) ** 2))
    avg_snr_improvement = np.mean(snr_improvements) if len(snr_improvements) > 0 else 1.0

    mean_error_m = mean_error * bin_to_meters
    median_error_m = median_error * bin_to_meters
    rmse_m = rmse * bin_to_meters
    dist_unc_m = np.mean(dist_uncertainties) * bin_to_meters if len(dist_uncertainties) else None

    metrics = {
        "avg_val_loss": avg_val_loss,
        "mean_error_bins": mean_error,
        "median_error_bins": median_error,
        "rmse_bins": rmse,
        "std_error_bins": std_error,
        "mae_bins": mae,
        "mean_error_m": mean_error_m,
        "median_error_m": median_error_m,
        "rmse_m": rmse_m,
        "snr_improvement": avg_snr_improvement,
        "dist_uncertainty_m": dist_unc_m,
    }
    return metrics

# ==========================================
# Plotting helpers
# ==========================================

def plot_metrics(distance_errors, snr_improvements, train_losses, model_name="model"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(distance_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"{model_name} Distance Error (bins)")
    plt.xlabel("Error (bins)")
    plt.ylabel("Freq")
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    if len(train_losses) > 1:
        plt.plot(train_losses[1:], marker='o')
        plt.title(f"{model_name} Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_metrics.png", dpi=150)
    plt.close()

# ==========================================
# Main
# ==========================================

def get_model(name, args):
    if name == "unet":
        return UNet1D(p_drop=0.0)
    if name == "mcd_unet":
        return UNet1D(p_drop=args.dropout)
    if name == "mlp":
        return NaiveMLP(L=1600)
    if name == "resnet":
        return ResNet1D()
    if name == "transformer":
        return Transformer1D(L=1600)
    if name == "lenet":
        return LeNet1D(L=1600)
    if name == "alexnet":
        return AlexNet1D(L=1600)
    if name == "tcn":
        return TCN1D(L=1600)
    if name == "mobilenet":
        return MobileNet1D(L=1600)
    if name == "performer":
        return Performer1D(L=1600)
    if name == "longformer":
        return Longformer1D(L=1600)
    raise ValueError(f"Unknown model {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Eval mode (no training)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Checkpoint path')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet','mcd_unet','mlp','resnet','transformer','lenet','alexnet','tcn','mobilenet','performer','longformer'],
                        help='Model to train/eval')
    parser.add_argument('--bench', action='store_true', help='Train/eval all baselines')
    parser.add_argument('--bench_additional', action='store_true', help='Train/eval all additional baselines')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout prob for MC U-Net')
    parser.add_argument('--mc_samples', type=int, default=50, help='MC forward passes')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--train_size', type=int, default=100000, help='Train samples')
    parser.add_argument('--val_size', type=int, default=10000, help='Val samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim = UnderwaterSPADSimulator(bin_resolution_ps=104.17, lidar_period_bins=1600)

    # Difficulty mixes
    train_mix = {"easy":0.3, "medium":0.4, "hard":0.3}
    val_mix   = {"easy":0.46, "medium":0.4, "hard":0.14}

    logger.info("Generating datasets...")
    train_dataset = build_dataset(sim, n=args.train_size, difficulty_mix=train_mix)
    val_dataset   = build_dataset(sim, n=args.val_size, difficulty_mix=val_mix)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.MSELoss()

    def run_one(model_name):
        logger.info("\n=== Running %s ===", model_name)
        model = get_model(model_name, args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        is_mc = (model_name == "mcd_unet")
        checkpoint_path = args.checkpoint.replace(".pth", f"_{model_name}.pth")

        if args.eval:
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                logger.info("Loaded checkpoint for %s.", model_name)
            else:
                logger.error("Checkpoint not found for %s, exiting.", model_name)
                return None

        train_losses = []
        if not args.eval:
            train_losses, val_losses, best_val, best_state = train_one(
                model, train_loader, val_loader, device,
                criterion, optimizer, args.epochs,
                is_mc=is_mc, mc_samples=args.mc_samples,
                checkpoint_path=checkpoint_path
            )
            if best_state is not None:
                model.load_state_dict(best_state)
            elif os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Restored best model (val loss %.8f) for evaluation.", best_val)

        metrics = evaluate(
            model, val_loader, device, criterion,
            is_mc=is_mc, mc_samples=args.mc_samples,
            bin_to_meters=0.5 * (3e8 / 1.33) * (104.17e-12)
        )

        logger.info("\nValidation Metrics for %s", model_name)
        for k,v in metrics.items():
            if v is None:
                continue
            logger.info("  %s: %.8f", k, v)
        return metrics, train_losses

    results = {}
    if args.bench:
        for mname in ["mlp","resnet","transformer","lenet","alexnet","tcn","mobilenet","performer","longformer","unet","mcd_unet"]:
            metrics, train_losses = run_one(mname)
            if metrics is not None:
                results[mname] = metrics
    elif args.bench_additional:
        for mname in ["alexnet","tcn","mobilenet","performer","longformer"]:
            metrics, train_losses = run_one(mname)
            if metrics is not None:
                results[mname] = metrics
    else:
        run_one(args.model)

    if args.bench:
        logger.info("\n=== BENCH SUMMARY (avg_val_loss) ===")
        for k,v in sorted(results.items(), key=lambda kv: kv[1]["avg_val_loss"]):
            logger.info("%s: %.8f", f"{k:12s}", v["avg_val_loss"])

if __name__ == "__main__":
    main()
