"""
ETEED: Enhanced Tiny and Efficient Edge Detector for SEM Metrology
===================================================================
Single-file consolidated implementation specifically designed for SEM
semiconductor metrology edge detection.

Key innovations over original TEED:
  1. ECA (Efficient Channel Attention) in skip connections — 0 extra params
  2. Optional Sobel gradient input channel — guides model toward edges
  3. Thin Edge Head — directly outputs single-pixel skeleton edges
     (trained with Zhang-Suen skeletonized ground truth)
  4. Focal Tversky + Dice loss — optimized for thin, clean edges
  5. Distance Field auxiliary head — predicts distance to nearest edge
     (ridges → single-pixel skeleton, no post-processing needed)

Model variants (controlled by config):
  - ETED: Enhanced TED backbone (TEED + attention + gradient input)
  - "dist" head: Predicts edge distance field (find ridges → skeleton)
  - "thin" head: Directly predicts single-pixel skeleton edges

Architecture: TEED backbone (3-stage CNN, ~58K params)
  + ECA attention modules
  + Thin Edge Head (3 conv layers, ~2K params)
  Total: ~65K parameters

Usage:
  python teed_sem.py --mode train --data_dir ./data
  python teed_sem.py --mode train --data_dir ./data --use_skeleton_gt
  python teed_sem.py --mode infer --image_path ./test.jpg
  python teed_sem.py --mode infer --image_dir ./sem_images/ --thin_output
  python teed_sem.py --mode export_onnx
  python teed_sem.py --mode info
"""

from __future__ import print_function

# ============================================================================
# IDE Direct Run Settings
# ============================================================================
IDE_MODE = "info"               # train | infer | export_onnx | info
IDE_DATA_DIR = "./data"         # Path to data/raw/ + data/mask/
IDE_EPOCHS = 10
IDE_LR = 8e-4
IDE_IMG_SIZE = 352
IDE_BATCH_SIZE = 8
IDE_USE_SKELETON_GT = True      # Train thin head with skeletonized GT
IDE_PRETRAINED = ""             # Optional: path to pretrained TEED weights
IDE_CHECKPOINT = "./checkpoints/best_model.pth"
IDE_RESULT_DIR = "./results"
IDE_IMAGE_PATH = ""
IDE_IMAGE_DIR = ""

import os
import sys
import time
import random
import math
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class SEMConfig:
    """Configuration for ETEED — SEM metrology edge detection."""

    # ── Paths ──────────────────────────────────────────────────────────
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    log_dir: str = "./logs"

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 10
    batch_size: int = 8
    lr: float = 8e-4
    wd: float = 2e-4
    img_size: int = 352
    num_workers: int = 4
    seed: int = 1021
    fp16: bool = False
    grad_clip: float = 0.0

    # ── Model architecture ─────────────────────────────────────────────
    use_attention: bool = True          # ECA attention in skip connections
    use_gradient_input: bool = True     # Sobel gradient as 4th input channel
    use_thin_head: bool = True          # Direct thin-edge prediction head
    use_dist_head: bool = False         # Distance field prediction head
    pretrained_backbone: str = ""       # Path to pretrained TEED weights

    # ── Loss configuration ─────────────────────────────────────────────
    loss_tversky_alpha: float = 0.7     # FP penalty (higher = fewer noise lines)
    loss_tversky_beta: float = 0.3      # FN penalty (higher = fewer missed edges)
    loss_tversky_gamma: float = 0.75    # Focal exponent
    loss_edge_weight: float = 1.0       # Weight for edge head loss
    loss_thin_weight: float = 1.0       # Weight for thin head loss
    loss_dist_weight: float = 0.5       # Weight for distance field loss

    # ── Validation split ───────────────────────────────────────────────
    val_split: float = 0.2

    # ── LR scheduler ───────────────────────────────────────────────────
    lr_scheduler: str = "step"
    lr_milestones: tuple = (4,)
    lr_gamma: float = 0.1
    lr_min: float = 1e-6
    lr_warmup_epochs: int = 0
    plateau_patience: int = 2
    plateau_factor: float = 0.5

    # ── Data augmentation ──────────────────────────────────────────────
    aug_rotation: float = 15.0
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_crop_prob: float = 0.4
    aug_crop_min_size: int = 256
    aug_edge_boost: float = 0.2        # Boost edge pixel values in GT

    # ── Mean pixel values for normalization (BGR) ──────────────────────
    mean_pixels: tuple = (103.939, 116.779, 123.68)

    # ── Training schedule ──────────────────────────────────────────────
    save_interval: int = 1
    val_interval: int = 1
    log_interval: int = 20
    viz_interval: int = 200
    early_stopping_patience: int = 0
    resume_from: str = ""

    # ── Inference ──────────────────────────────────────────────────────
    predict_all_outputs: bool = False
    thin_output: bool = True           # Output thin edges directly

    # ── ONNX export ────────────────────────────────────────────────────
    onnx_input_size: tuple = (1, 3, 480, 480)
    onnx_opset: int = 14
    onnx_dynamic_axes: bool = True


# ============================================================================
# 2. ACTIVATION FUNCTION (Smish)
# ============================================================================

def smish(input: torch.Tensor) -> torch.Tensor:
    """Smish: input * tanh(ln(1 + sigmoid(input)))"""
    return input * torch.tanh(torch.log(1.0 + torch.sigmoid(input)))


class Smish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return smish(input)


Fsmish = smish  # Used by DoubleFusion


# ============================================================================
# 3. ATTENTION MODULE (ECA — Efficient Channel Attention)
# ============================================================================

class ECA(nn.Module):
    """
    Efficient Channel Attention.
    Adds ~0 parameters. Uses adaptive 1D convolution kernel size.
    Reference: Wang et al., "ECA-Net", CVPR 2020.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        kernel_size = max(kernel_size, 3)  # Minimum kernel size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        # Global Average Pooling → (B, C)
        y = x.mean(dim=(2, 3), keepdim=False)  # (B, C)
        # 1D conv along channel dimension: (B, C) → (B, 1, C) for Conv1d
        y = self.conv(y.unsqueeze(1))  # Conv1d(1,1,k): (B, 1, C) → (B, 1, C)
        y = y.squeeze(1)  # (B, C)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# ============================================================================
# 4. ETED MODEL ARCHITECTURE
# ============================================================================

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class DoubleFusion(nn.Module):
    """TED fusion layer (DWconv + PixelShuffle)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3,
                                 stride=1, padding=1, groups=in_ch)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24, kernel_size=3,
                                 stride=1, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return Fsmish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features: int, out_features: int):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module('smish1', Smish())
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True))

    def forward(self, x):
        x1, x2 = x
        new_features = super().forward(Fsmish(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, input_features: int, out_features: int):
        super().__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(input_features, out_features))
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features: int, up_scale: int):
        super().__init__()
        self.constant_features = 16
        layers = self._make_deconv_layers(in_features, up_scale)
        self.features = nn.Sequential(*layers)

    def _make_deconv_layers(self, in_features: int, up_scale: int):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = 1 if i == up_scale - 1 else self.constant_features
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, stride: int, use_ac: bool = False):
        super().__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        if self.use_ac:
            self.smish = Smish()

    def forward(self, x):
        x = self.conv(x)
        return self.smish(x) if self.use_ac else x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features: int, mid_features: int,
                 out_features: int = None, stride: int = 1, use_act: bool = True):
        super().__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class ThinEdgeHead(nn.Module):
    """
    Thin Edge Prediction Head.
    Takes the fused multi-scale features + coarse edge probability,
    predicts a single-pixel wide edge skeleton.

    Trained with skeletonized ground truth (Zhang-Suen applied to GT edges).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # in_channels = fused_features channels (3 from cat of out1,out2,out3)
        # We also take the coarse edge prob as additional input
        self.conv1 = nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)
        self.smish = Smish()

    def forward(self, fused_features, coarse_edge):
        """
        Args:
            fused_features: (B, 3, H, W) — cat of out1, out2, out3
            coarse_edge: (B, 1, H, W) — sigmoid-activated edge probability
        Returns:
            thin_logits: (B, 1, H, W)
        """
        x = torch.cat([fused_features, coarse_edge], dim=1)
        x = self.smish(self.conv1(x))
        x = self.smish(self.conv2(x))
        x = self.conv3(x)
        return x


class DistHead(nn.Module):
    """
    Distance Field Prediction Head.
    Predicts the distance from each pixel to the nearest edge.
    Edges are at the local minima of this field.

    Output is in [0, 1] where 0 = on edge, 1 = far from edge.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)
        self.smish = Smish()

    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, 3, H, W) — cat of out1, out2, out3
        Returns:
            dist: (B, 1, H, W) — distance field [0, 1]
        """
        x = self.smish(self.conv1(fused_features))
        x = self.smish(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


class ETED(nn.Module):
    """
    Enhanced Tiny and Efficient Edge Detector for SEM Metrology.

    Improvements over original TEED:
    - ECA attention after each block for better feature selection
    - Optional gradient input (4-channel input: RGB + Sobel magnitude)
    - Thin Edge Head: directly predicts single-pixel skeleton edges
    - Distance Field Head: predicts edge distance field (ridges → skeleton)
    - All heads are optional, configurable via SEMConfig
    """

    def __init__(self, cfg: SEMConfig = None):
        super().__init__()
        if cfg is None:
            cfg = SEMConfig()

        in_channels = 4 if cfg.use_gradient_input else 3
        self.use_gradient_input = cfg.use_gradient_input
        self.use_attention = cfg.use_attention
        self.use_thin_head = cfg.use_thin_head
        self.use_dist_head = cfg.use_dist_head

        # ── TEED Backbone ──────────────────────────────────────────
        self.block_1 = DoubleConvBlock(in_channels, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        # ── Attention modules ──────────────────────────────────────
        if self.use_attention:
            self.attn_1 = ECA(16)
            self.attn_2 = ECA(32)
            self.attn_3 = ECA(48)

        # ── Fusion layer ───────────────────────────────────────────
        self.block_cat = DoubleFusion(3, 3)

        # ── Additional heads ───────────────────────────────────────
        if self.use_thin_head:
            self.thin_head = ThinEdgeHead(in_channels=3)
        if self.use_dist_head:
            self.dist_head = DistHead(in_channels=3)

        self.apply(weight_init)

    def _compute_gradient_channel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Sobel gradient magnitude as an additional input channel.
        Applied per-image in the batch, using simple 3x3 Sobel operators.
        """
        # Define Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=x.device).view(1, 1, 3, 3)

        b, c, h, w = x.shape
        # Convert to grayscale (standard weights)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Normalize to [0, 1]
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        return grad_mag

    def resize_input(self, tensor: torch.Tensor) -> torch.Tensor:
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w = ((t_shape[3] // 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            return F.interpolate(tensor, size=(img_h, img_w),
                                 mode='bilinear', align_corners=False)
        return tensor

    def slice_tensor(self, tensor: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = target_shape
        if tensor.shape[2] != h or tensor.shape[3] != w:
            return F.interpolate(tensor, size=(h, w),
                                 mode='bilinear', align_corners=False)
        return tensor

    def forward(self, x: torch.Tensor, return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) or (B, 4, H, W) if gradient channel pre-computed
            return_all: if True, return all intermediate outputs

        Returns:
            dict with keys: 'edge' (always), 'thin' (if thin_head enabled),
            'dist' (if dist_head enabled), 'out1','out2','out3' (if return_all)
        """
        assert x.ndim == 4, x.shape
        original_shape = x.shape[2:]

        # Preprocess: add gradient channel if configured
        if self.use_gradient_input and x.shape[1] == 3:
            grad = self._compute_gradient_channel(x)
            x = torch.cat([x, grad], dim=1)

        # Resize to multiple of 8
        x = self.resize_input(x)

        # ── Block 1 ──────────────────────────────────────────────
        block_1 = self.block_1(x)                                    # (B,16,H/2,W/2)
        block_1 = self.attn_1(block_1) if self.use_attention else block_1
        block_1_side = self.side_1(block_1)                          # (B,32,H/4,W/4)

        # ── Block 2 ──────────────────────────────────────────────
        block_2 = self.block_2(block_1)                              # (B,32,H/2,W/2)
        block_2 = self.attn_2(block_2) if self.use_attention else block_2
        block_2_down = self.maxpool(block_2)                         # (B,32,H/4,W/4)
        block_2_add = block_2_down + block_1_side                    # (B,32,H/4,W/4)

        # ── Block 3 ──────────────────────────────────────────────
        block_3_pre = self.pre_dense_3(block_2_down)                 # (B,48,H/4,W/4)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre])      # (B,48,H/4,W/4)
        block_3 = self.attn_3(block_3) if self.use_attention else block_3

        # ── Upsampling ───────────────────────────────────────────
        out_1 = self.up_block_1(block_1)                             # (B,1,H,W)
        out_2 = self.up_block_2(block_2)                             # (B,1,H,W)
        out_3 = self.up_block_3(block_3)                             # (B,1,H,W)

        # ── Multi-scale fusion ───────────────────────────────────
        fused_cat = torch.cat([out_1, out_2, out_3], dim=1)         # (B,3,H,W)
        edge = self.block_cat(fused_cat)                             # (B,1,H,W)

        # Crop to original size
        out_1 = self.slice_tensor(out_1, original_shape)
        out_2 = self.slice_tensor(out_2, original_shape)
        out_3 = self.slice_tensor(out_3, original_shape)
        edge = self.slice_tensor(edge, original_shape)
        fused_cat = self.slice_tensor(fused_cat, original_shape)

        outputs = {'edge': edge, 'out1': out_1, 'out2': out_2, 'out3': out_3}

        # ── Thin Edge Head ───────────────────────────────────────
        if self.use_thin_head:
            coarse_prob = torch.sigmoid(edge)
            thin_logits = self.thin_head(fused_cat, coarse_prob)
            outputs['thin'] = thin_logits

        # ── Distance Field Head ──────────────────────────────────
        if self.use_dist_head:
            dist = self.dist_head(fused_cat)
            outputs['dist'] = dist

        if not return_all:
            del outputs['out1'], outputs['out2'], outputs['out3']

        return outputs


# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

def focal_tversky_loss(pred: torch.Tensor, target: torch.Tensor,
                       alpha: float = 0.7, beta: float = 0.3,
                       gamma: float = 0.75, smooth: float = 1e-8) -> torch.Tensor:
    """
    Focal Tversky Loss for edge detection.

    Tversky = (TP + s) / (TP + α·FP + β·FN + s)
    Focal Tversky = (1 - Tversky)^γ

    α=0.7 → strong penalty on false positives (noise lines)
    β=0.3 → moderate penalty on false negatives (missed edges)
    γ=0.75 → focus on hard examples
    """
    pred = torch.sigmoid(pred)
    b = pred.shape[0]
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    loss = torch.pow(1 - tversky, gamma)
    return loss.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-8) -> torch.Tensor:
    """
    Dice Loss — good for highly imbalanced data (skeleton edges are ~0.1% of pixels).
    """
    pred = torch.sigmoid(pred)
    b = pred.shape[0]
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def distance_field_loss(pred_dist: torch.Tensor, target_edges: torch.Tensor,
                        smooth: float = 1e-8) -> torch.Tensor:
    """
    Distance field regression loss.

    For each pixel, compute its Euclidean distance to the nearest edge pixel
    in the target. The model predicts this distance (normalized to [0, 1]).

    Uses L1 loss weighted toward near-edge pixels (more important for metrology).
    """
    b, _, h, w = pred_dist.shape
    loss_total = 0.0

    for i in range(b):
        edge_mask = target_edges[i, 0] > 0.5

        if edge_mask.sum() == 0:
            continue

        # Compute ground truth distance field using distance transform
        gt_dist_np = cv2.distanceTransform(
            (1 - edge_mask.float().cpu().numpy().astype(np.uint8)),
            cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Normalize
        max_dist = gt_dist_np.max()
        if max_dist > 0:
            gt_dist_np = gt_dist_np / max_dist

        gt_dist = torch.from_numpy(gt_dist_np).float().to(pred_dist.device)

        # Weighted L1: higher weight for pixels near edges
        near_edge_weight = 1.0 / (gt_dist + 0.1)  # Higher weight closer to edge
        diff = torch.abs(pred_dist[i, 0] - gt_dist)
        loss_total += (diff * near_edge_weight).mean()

    return loss_total / max(b, 1)


def bdcn_loss2(inputs: torch.Tensor, targets: torch.Tensor,
               l_weight: float = 1.1) -> torch.Tensor:
    """BDCN loss v2 — class-balanced BCE for multi-scale supervision."""
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float())
    num_negative = torch.sum((mask <= 0.0).float())

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs = torch.sigmoid(inputs)
    cost = F.binary_cross_entropy(inputs, targets.float(), weight=mask.detach(),
                                  reduction='none')
    return cost.float().mean((1, 2, 3)).sum()


def combined_sem_loss(outputs: Dict[str, torch.Tensor], target: torch.Tensor,
                      skeleton_target: torch.Tensor = None,
                      cfg: SEMConfig = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined loss for SEM edge detection.

    - Edge head: Focal Tversky loss
    - Thin head: Dice loss (on skeletonized GT)
    - Dist head: Distance field L1 loss
    - Multi-scale outputs (out1, out2, out3): BDCN loss

    Args:
        outputs: model forward output dict
        target: ground truth edge map (thick, as in BIPED)
        skeleton_target: skeletonized GT for thin head supervision
        cfg: SEMConfig
    """
    if cfg is None:
        cfg = SEMConfig()

    loss_dict = {}

    # Edge head loss (Focal Tversky)
    loss_edge = focal_tversky_loss(
        outputs['edge'], target,
        alpha=cfg.loss_tversky_alpha,
        beta=cfg.loss_tversky_beta,
        gamma=cfg.loss_tversky_gamma)
    loss_dict['edge'] = loss_edge.item()
    total = cfg.loss_edge_weight * loss_edge

    # Thin head loss (Dice on skeleton GT)
    if 'thin' in outputs and skeleton_target is not None:
        loss_thin = dice_loss(outputs['thin'], skeleton_target)
        loss_dict['thin'] = loss_thin.item()
        total = total + cfg.loss_thin_weight * loss_thin

    # Distance field head loss
    if 'dist' in outputs:
        loss_dist = distance_field_loss(outputs['dist'], target)
        loss_dict['dist'] = loss_dist.item()
        total = total + cfg.loss_dist_weight * loss_dist

    # Multi-scale BDCN loss
    l_weights = [1.1, 0.7, 1.1, 1.3]
    for i, key in enumerate(['out1', 'out2', 'out3']):
        if key in outputs:
            loss_ms = bdcn_loss2(outputs[key], target, l_weights[i])
            loss_dict[key] = loss_ms.item()
            total = total + 0.5 * loss_ms  # 0.5 weight for multi-scale terms

    return total, loss_dict


# ============================================================================
# 6. DATASET
# ============================================================================

class SEMEdgeDataset(Dataset):
    """
    Dataset for SEM edge detection training.

    Expects:
        data/raw/    - source images (SEM or any)
        data/mask/   - edge maps (grayscale, white edges on black background)

    If use_skeleton_gt=True: also computes Zhang-Suen skeletonized GT
    for the thin edge head supervision.
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, data_dir: str, cfg: SEMConfig, train_mode: bool = True,
                 use_skeleton_gt: bool = False):
        self.cfg = cfg
        self.train_mode = train_mode
        self.use_skeleton_gt = use_skeleton_gt

        raw_dir = os.path.join(data_dir, 'raw')
        mask_dir = os.path.join(data_dir, 'mask')

        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"raw directory not found: {raw_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"mask directory not found: {mask_dir}")

        raw_files = sorted([f for f in os.listdir(raw_dir)
                           if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS])
        mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
                      if os.path.splitext(f)[1].lower() in self.IMG_EXTENSIONS}

        self.samples = []
        for rf in raw_files:
            stem = os.path.splitext(rf)[0]
            if stem in mask_files:
                self.samples.append((
                    os.path.join(raw_dir, rf),
                    os.path.join(mask_dir, mask_files[stem]),
                    stem
                ))
            else:
                print(f"[WARN] No matching mask for {rf}, skipping.")

        if len(self.samples) == 0:
            raise RuntimeError(f"No image-mask pairs found in {data_dir}")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")
        if use_skeleton_gt:
            print(f"[Dataset] Skeleton GT will be computed on-the-fly for thin head")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, stem = self.samples[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"Cannot read {img_path} or {mask_path}")

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask /= 255.0

        if self.train_mode:
            img, mask = self._augment(img, mask)

        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size),
                          interpolation=cv2.INTER_NEAREST)

        mean_bgr = np.array(self.cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
        img -= mean_bgr
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask[np.newaxis, ...].copy()).float()

        result = {
            'images': img_t,
            'labels': mask_t,
            'file_name': stem + '.png'
        }

        # Compute skeleton GT for thin head
        if self.use_skeleton_gt:
            skeleton = zhang_suen_thinning(mask_t[0].numpy())
            result['skeleton'] = torch.from_numpy(skeleton[np.newaxis, ...]).float()

        return result

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        h, w = mask.shape

        # Random crop
        if cfg.aug_crop_prob > 0 and random.random() < cfg.aug_crop_prob:
            min_sz = min(cfg.aug_crop_min_size, h, w)
            crop_h = random.randint(min_sz, h)
            crop_w = random.randint(min_sz, w)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img = img[top:top + crop_h, left:left + crop_w]
            mask = mask[top:top + crop_h, left:left + crop_w]

        # Rotation
        if cfg.aug_rotation > 0 and random.random() < 0.5:
            angle = random.uniform(-cfg.aug_rotation, cfg.aug_rotation)
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        # Flip
        if cfg.aug_hflip and random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if cfg.aug_vflip and random.random() < 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # Color jitter
        if cfg.aug_brightness > 0 or cfg.aug_contrast > 0:
            delta_b = random.uniform(-cfg.aug_brightness * 255,
                                     cfg.aug_brightness * 255)
            img = img + delta_b
            alpha_c = 1.0 + random.uniform(-cfg.aug_contrast, cfg.aug_contrast)
            mean_img = img.mean()
            img = (img - mean_img) * alpha_c + mean_img

        # Edge boost
        if cfg.aug_edge_boost > 0:
            mask[mask > 0.1] += cfg.aug_edge_boost
            mask = np.clip(mask, 0.0, 1.0)

        return img, mask


# ============================================================================
# 7. SKELETONIZATION (Zhang-Suen for GT preparation)
# ============================================================================

def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen skeletonization for ground truth preparation.
    Converts thick edge maps to single-pixel skeletons.

    Args:
        binary: (H, W) float/uint8 edge map
    Returns:
        skeleton: (H, W) uint8 single-pixel skeleton
    """
    img = (binary > 0.1).astype(np.uint8)
    h, w = img.shape
    skeleton = img.copy()

    def _neighbors(y, x):
        n = []
        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)]:
            ny, nx = y + dy, x + dx
            n.append(skeleton[ny, nx] if 0 <= ny < h and 0 <= nx < w else 0)
        return n

    def _transitions(n):
        t = 0
        for i in range(8):
            if n[i] == 0 and n[(i + 1) % 8] == 1:
                t += 1
        return t

    changed = True
    while changed:
        changed = False
        # Step 1
        to_remove = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue
                n = _neighbors(y, x)
                s = sum(n)
                if not (2 <= s <= 6):
                    continue
                if _transitions(n) != 1:
                    continue
                if n[0] * n[2] * n[4] != 0:
                    continue
                if n[2] * n[4] * n[6] != 0:
                    continue
                to_remove.append((y, x))
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True

        # Step 2
        to_remove = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue
                n = _neighbors(y, x)
                s = sum(n)
                if not (2 <= s <= 6):
                    continue
                if _transitions(n) != 1:
                    continue
                if n[0] * n[2] * n[6] != 0:
                    continue
                if n[0] * n[4] * n[6] != 0:
                    continue
                to_remove.append((y, x))
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True

    return skeleton


# ============================================================================
# 8. UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def image_normalization(img: np.ndarray, img_min: float = 0, img_max: float = 255,
                        epsilon: float = 1e-12) -> np.ndarray:
    img = np.float32(img)
    mn, mx = np.min(img), np.max(img)
    return (img - mn) * (img_max - img_min) / (mx - mn + epsilon) + img_min


def save_edge_output(tensor: torch.Tensor, output_dir: str, file_name: str,
                     original_shape: Tuple[int, int] = None,
                     is_thin: bool = False) -> None:
    """Save edge prediction as image."""
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        if is_thin:
            tensor = torch.sigmoid(tensor)
        tensor = tensor.cpu().detach().numpy()
    img = np.squeeze(tensor)
    img = np.uint8(image_normalization(img))
    img = cv2.bitwise_not(img)
    if original_shape is not None:
        img = cv2.resize(img, (original_shape[1], original_shape[0]),
                         interpolation=cv2.INTER_NEAREST if is_thin else cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, file_name), img)


# ============================================================================
# 9. TRAINING
# ============================================================================

def create_optimizer(model: nn.Module, cfg: SEMConfig) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)


def create_scheduler(optimizer: optim.Optimizer, cfg: SEMConfig,
                     steps_per_epoch: int):
    name = cfg.lr_scheduler.lower()
    total_steps = cfg.epochs * steps_per_epoch

    if name == "step":
        if cfg.lr_milestones:
            return optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[m * steps_per_epoch for m in cfg.lr_milestones],
                gamma=cfg.lr_gamma)
        return optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch,
                                         gamma=cfg.lr_gamma)
    elif name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=cfg.lr_min)
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.plateau_factor,
            patience=cfg.plateau_patience, min_lr=cfg.lr_min)
    elif name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=10000.0)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(epoch: int, dataloader: DataLoader, model: ETED,
                    optimizer: optim.Optimizer, scheduler, device: torch.device,
                    scaler, cfg: SEMConfig, use_skeleton: bool = False):
    model.train()
    total_loss = 0.0
    loss_components = {}

    for batch_id, sample in enumerate(dataloader):
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)
        skeleton = sample.get('skeleton', None)
        if skeleton is not None:
            skeleton = skeleton.to(device)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            outputs = model(images, return_all=True)
            loss, loss_dict = combined_sem_loss(
                outputs, labels,
                skeleton_target=skeleton if use_skeleton else None,
                cfg=cfg)

        optimizer.zero_grad()
        if cfg.fp16:
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if cfg.lr_scheduler in ("step", "cosine", "onecycle"):
            scheduler.step()

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0.0) + v

        if batch_id % cfg.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            comp_str = ' | '.join(f'{k}:{v:.4f}' for k, v in loss_dict.items())
            print(f"  [E{epoch:3d} B{batch_id:4d}/{len(dataloader):4d}] "
                  f"Loss:{loss.item():.4f} ({comp_str}) LR:{lr:.2e}")

    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    return avg_loss, avg_components


@torch.no_grad()
def validate_one_epoch(epoch: int, dataloader: DataLoader, model: ETED,
                       device: torch.device, cfg: SEMConfig,
                       use_skeleton: bool = False):
    model.eval()
    total_loss = 0.0

    result_dir = os.path.join(cfg.checkpoint_dir, 'val_results', f'epoch_{epoch:03d}')
    os.makedirs(result_dir, exist_ok=True)

    for batch_id, sample in enumerate(dataloader):
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)
        file_names = sample.get('file_name', [f'{batch_id}.png'])
        skeleton = sample.get('skeleton', None)
        if skeleton is not None:
            skeleton = skeleton.to(device)

        outputs = model(images, return_all=True)
        loss, _ = combined_sem_loss(outputs, labels,
                                    skeleton_target=skeleton if use_skeleton else None,
                                    cfg=cfg)
        total_loss += loss.item()

        if batch_id < 5:
            if isinstance(file_names, (list, tuple)):
                fn = file_names[0] if len(file_names) > 0 else f'{batch_id}.png'
            else:
                fn = str(file_names)
            save_edge_output(outputs['edge'][0], result_dir, fn)
            if 'thin' in outputs:
                save_edge_output(outputs['thin'][0],
                                 os.path.join(result_dir, 'thin'), fn, is_thin=True)

    return total_loss / len(dataloader)


def train(cfg: SEMConfig):
    """Main training function."""
    print("=" * 60)
    print("ETEED Training — SEM Metrology Edge Detection")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    use_skeleton = cfg.use_thin_head

    # Dataset
    full_dataset = SEMEdgeDataset(cfg.data_dir, cfg, train_mode=True,
                                  use_skeleton_gt=use_skeleton)
    n_val = max(1, int(len(full_dataset) * cfg.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(cfg.seed))

    # Subset wrappers
    class SubsetDS:
        def __init__(self, full_ds, indices, train_mode, use_skel):
            self.full_ds = full_ds
            self.indices = indices
            self.train_mode = train_mode
            self.use_skel = use_skel
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            old_mode = self.full_ds.train_mode
            self.full_ds.train_mode = self.train_mode
            self.full_ds.use_skeleton_gt = self.use_skel
            item = self.full_ds[self.indices[idx]]
            self.full_ds.train_mode = old_mode
            return item

    train_ds = SubsetDS(full_dataset, train_ds.indices, True, use_skeleton)
    val_ds = SubsetDS(full_dataset, val_ds.indices, False, use_skeleton)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # Model
    model = ETED(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Load pretrained backbone if specified
    if cfg.pretrained_backbone and os.path.isfile(cfg.pretrained_backbone):
        print(f"Loading pretrained backbone: {cfg.pretrained_backbone}")
        ckpt = torch.load(cfg.pretrained_backbone, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            backbone_state = ckpt['model_state_dict']
        else:
            backbone_state = ckpt
        # Load only backbone weights (ignore new heads)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in backbone_state.items()
                          if k in model_dict and 'thin_head' not in k and 'dist_head' not in k
                          and 'attn' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(pretrained_dict)} backbone layers")

    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    start_epoch = 0
    best_val_loss = float('inf')
    early_stop_counter = 0

    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        print(f"Resuming from: {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('best_val_loss', float('inf'))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n{'─' * 50}")
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        print(f"{'─' * 50}")

        train_loss, loss_comps = train_one_epoch(
            epoch, train_loader, model, optimizer, scheduler, device, scaler,
            cfg, use_skeleton)

        comp_str = ' | '.join(f'{k}:{v:.4f}' for k, v in loss_comps.items())
        print(f"  Train Loss: {train_loss:.4f} ({comp_str})")

        if cfg.val_interval > 0 and (epoch + 1) % cfg.val_interval == 0:
            val_loss = validate_one_epoch(epoch, val_loader, model, device,
                                          cfg, use_skeleton)
            print(f"  Val Loss: {val_loss:.4f}")

            if cfg.lr_scheduler == "plateau":
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': cfg,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
                print(f"  >>> Best model saved (val_loss: {val_loss:.4f})")
            else:
                early_stop_counter += 1

        if cfg.save_interval > 0 and (epoch + 1) % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': cfg,
            }, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth'))

        if cfg.early_stopping_patience > 0 and early_stop_counter >= cfg.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}!")
            break

    # Save final model
    torch.save({
        'epoch': cfg.epochs,
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }, os.path.join(cfg.checkpoint_dir, 'final_model.pth'))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ============================================================================
# 10. INFERENCE
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device,
               cfg: SEMConfig = None) -> ETED:
    """Load ETEED model from checkpoint, handling channel mismatches."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect checkpoint format
    if isinstance(ckpt, dict) and 'config' in ckpt:
        cfg = ckpt['config']
    elif cfg is None:
        cfg = SEMConfig()

    model = ETED(cfg).to(device)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Handle channel mismatch: pretrained weights may be 3-channel,
    # but model expects 4-channel (RGB + gradient)
    adapt_state_dict(state_dict, model, cfg)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model loaded: {checkpoint_path}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Config: attn={cfg.use_attention}, grad_input={cfg.use_gradient_input}, "
          f"thin_head={cfg.use_thin_head}, dist_head={cfg.use_dist_head}")
    return model


def adapt_state_dict(state_dict: dict, model: nn.Module, cfg: SEMConfig):
    """
    Adapt pretrained weights for channel count mismatch.
    If pretrained has 3 input channels but model expects 4 (RGB + gradient),
    initialize the 4th channel as the mean of the first 3.
    """
    first_conv_key = 'block_1.conv1.weight'
    if first_conv_key in state_dict and cfg.use_gradient_input:
        pretrained_weight = state_dict[first_conv_key]  # (16, 3, 3, 3)
        model_weight = model.state_dict()[first_conv_key]  # (16, 4, 3, 3)
        if pretrained_weight.shape[1] != model_weight.shape[1]:
            # Expand: new channel = mean of RGB channels
            new_weight = torch.zeros_like(model_weight)
            new_weight[:, :3] = pretrained_weight
            new_weight[:, 3:4] = pretrained_weight.mean(dim=1, keepdim=True)
            state_dict[first_conv_key] = new_weight
            print(f"  Adapted {first_conv_key}: {pretrained_weight.shape[1]}ch → "
                  f"{model_weight.shape[1]}ch")


def infer_single(model: ETED, image_path: str, output_dir: str,
                 device: torch.device, cfg: SEMConfig):
    """Inference on a single image."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    original_shape = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]

    img_f = img.astype(np.float32)
    mean_bgr = np.array(cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
    img_f -= mean_bgr
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t, return_all=cfg.predict_all_outputs)

    # Save edge probability map
    save_edge_output(outputs['edge'][0], output_dir, f'{stem}_edge.png', original_shape)

    # Save thin edge output
    if 'thin' in outputs and cfg.thin_output:
        thin_dir = os.path.join(output_dir, 'thin')
        save_edge_output(outputs['thin'][0], thin_dir, f'{stem}_thin.png',
                         original_shape, is_thin=True)

    # Save distance field
    if 'dist' in outputs:
        dist = outputs['dist'][0, 0].cpu().numpy()
        dist = np.uint8((1 - dist) * 255)  # Invert: edge = white
        if dist.shape != original_shape:
            dist = cv2.resize(dist, (original_shape[1], original_shape[0]))
        cv2.imwrite(os.path.join(output_dir, f'{stem}_dist.png'), dist)

    print(f"Results saved to {output_dir}/")


def infer_batch(model: ETED, image_dir: str, output_dir: str,
                device: torch.device, cfg: SEMConfig):
    """Inference on a directory of images."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    image_files = sorted([f for f in os.listdir(image_dir)
                         if os.path.splitext(f)[1].lower() in exts])

    if not image_files:
        print(f"No images in {image_dir}")
        return

    print(f"Processing {len(image_files)} images...")
    for i, fname in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {fname}")
        infer_single(model, os.path.join(image_dir, fname), output_dir, device, cfg)


# ============================================================================
# 11. ONNX EXPORT
# ============================================================================

def export_onnx(model: ETED, cfg: SEMConfig):
    """Export model to ONNX format."""
    onnx_path = os.path.join(cfg.checkpoint_dir, 'model.onnx')
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(*cfg.onnx_input_size)

    # Use a wrapper for ONNX export (only edge output)
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)['edge']

    wrapper = ONNXWrapper(model)

    dynamic_axes = None
    if cfg.onnx_dynamic_axes:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'edge': {0: 'batch', 2: 'height', 3: 'width'},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy_input, onnx_path,
            input_names=['input'],
            output_names=['edge'],
            dynamic_axes=dynamic_axes,
            opset_version=cfg.onnx_opset,
            do_constant_folding=True,
        )

    print(f"ONNX model exported: {onnx_path}")


# ============================================================================
# 12. MODEL INFO
# ============================================================================

def print_model_info(cfg: SEMConfig):
    """Print model architecture and configuration info."""
    print("=" * 60)
    print("ETEED — Enhanced TEED for SEM Metrology")
    print("=" * 60)

    model = ETED(cfg)
    params = count_parameters(model)

    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    model.eval()
    with torch.no_grad():
        outputs = model(dummy)

    print(f"\nParameters: {params:,}")
    print(f"Input: (1, 3, {cfg.img_size}, {cfg.img_size})")
    print(f"  → {'4-channel' if cfg.use_gradient_input else '3-channel'} input")
    print(f"  → ECA Attention: {'ON' if cfg.use_attention else 'OFF'}")
    for k, v in outputs.items():
        print(f"  Output '{k}': {list(v.shape)}")
    print(f"  → Thin Head: {'ON' if cfg.use_thin_head else 'OFF'}")
    print(f"  → Dist Head: {'ON' if cfg.use_dist_head else 'OFF'}")

    print(f"\nMemory: {params * 4 / 1024**2:.1f} MB (fp32)")

    print(f"\n{'─' * 40}")
    print("Configuration:")
    print(f"{'─' * 40}")
    for field_name in SEMConfig.__dataclass_fields__:
        value = getattr(cfg, field_name)
        if not isinstance(value, (str, int, float, bool, tuple)):
            continue
        print(f"  {field_name}: {value}")


# ============================================================================
# 13. CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ETEED — Enhanced TEED for SEM Metrology Edge Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python teed_sem.py --mode train --data_dir ./data
  python teed_sem.py --mode train --data_dir ./data --use_skeleton_gt --epochs 20
  python teed_sem.py --mode infer --image_path ./test.jpg --thin_output
  python teed_sem.py --mode infer --image_dir ./sem_images/ --thin_output
  python teed_sem.py --mode info
        """)

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'infer', 'export_onnx', 'info'],
                        help='Operation mode')

    # Paths
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1021)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--grad_clip', type=float, default=0.0)

    # Model
    parser.add_argument('--use_attention', action='store_true', default=True)
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable ECA attention')
    parser.add_argument('--use_gradient_input', action='store_true', default=True)
    parser.add_argument('--no_gradient_input', action='store_true',
                        help='Disable gradient input channel')
    parser.add_argument('--use_thin_head', action='store_true', default=True)
    parser.add_argument('--no_thin_head', action='store_true',
                        help='Disable thin edge head')
    parser.add_argument('--use_dist_head', action='store_true', default=False)
    parser.add_argument('--use_skeleton_gt', action='store_true', default=False,
                        help='Use skeletonized GT for thin head training')
    parser.add_argument('--pretrained_backbone', type=str, default='',
                        help='Path to pretrained TEED weights')

    # Loss
    parser.add_argument('--tversky_alpha', type=float, default=0.7)
    parser.add_argument('--tversky_beta', type=float, default=0.3)

    # LR scheduler
    parser.add_argument('--lr_scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'onecycle'])

    # Schedule
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--early_stopping_patience', type=int, default=0)
    parser.add_argument('--resume_from', type=str, default='')

    # Inference
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--image_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--thin_output', action='store_true', default=True)
    parser.add_argument('--predict_all_outputs', action='store_true', default=False)

    args = parser.parse_args()

    # Build config
    cfg = SEMConfig()
    for key, value in vars(args).items():
        if key in SEMConfig.__dataclass_fields__:
            setattr(cfg, key, value)

    # Handle negation flags
    if args.no_attention:
        cfg.use_attention = False
    if args.no_gradient_input:
        cfg.use_gradient_input = False
    if args.no_thin_head:
        cfg.use_thin_head = False

    return args, cfg


def main():
    args, cfg = parse_args()

    if args.mode == 'info':
        print_model_info(cfg)
        return

    if args.mode == 'train':
        train(cfg)

    elif args.mode == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(args.checkpoint, device, cfg)

        if args.image_path:
            infer_single(model, args.image_path, cfg.result_dir, device, cfg)
        elif args.image_dir:
            infer_batch(model, args.image_dir, cfg.result_dir, device, cfg)
        else:
            print("Error: --image_path or --image_dir required for infer mode.")
            sys.exit(1)

    elif args.mode == 'export_onnx':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model(args.checkpoint, device, cfg)
        export_onnx(model, cfg)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("=" * 60)
        print("ETEED — Running in IDE mode")
        print(f"  Current IDE_MODE: {IDE_MODE}")
        print("  Modify IDE_* variables at top of file to configure.")
        print("=" * 60)

        sys.argv.append("--mode")
        sys.argv.append(IDE_MODE)

        if IDE_MODE == "train":
            sys.argv.extend(["--data_dir", IDE_DATA_DIR])
            sys.argv.extend(["--epochs", str(IDE_EPOCHS)])
            sys.argv.extend(["--lr", str(IDE_LR)])
            sys.argv.extend(["--img_size", str(IDE_IMG_SIZE)])
            sys.argv.extend(["--batch_size", str(IDE_BATCH_SIZE)])
            sys.argv.extend(["--checkpoint_dir", "./checkpoints"])
            if IDE_USE_SKELETON_GT:
                sys.argv.append("--use_skeleton_gt")
            if IDE_PRETRAINED:
                sys.argv.extend(["--pretrained_backbone", IDE_PRETRAINED])

        elif IDE_MODE == "infer":
            sys.argv.extend(["--checkpoint", IDE_CHECKPOINT])
            sys.argv.extend(["--result_dir", IDE_RESULT_DIR])
            if IDE_IMAGE_PATH:
                sys.argv.extend(["--image_path", IDE_IMAGE_PATH])
            if IDE_IMAGE_DIR:
                sys.argv.extend(["--image_dir", IDE_IMAGE_DIR])

        elif IDE_MODE == "export_onnx":
            sys.argv.extend(["--checkpoint", IDE_CHECKPOINT])
            sys.argv.extend(["--checkpoint_dir", "./checkpoints"])

    main()
