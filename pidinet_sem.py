"""
PiDiNet-SEM: Pixel Difference Network for SEM Metrology Edge Detection
========================================================================
Single-file self-contained implementation inspired by PiDiNet (ICCV 2021).

Core innovation: Central Difference Convolution (CDC) — a drop-in replacement
for standard convolution that simultaneously captures intensity AND gradient
information. This naturally produces thin, localized edge responses without
post-processing.

Why CDC matters for thin edges:
  Standard Conv: y = Σ w_i · x_i         → captures intensity patterns
  Difference Conv: y = Σ w_i · (x_i - x_center)  → captures gradient patterns
  CDC: y = θ · vanilla + (1-θ) · difference       → best of both

The θ parameter is learned per-layer, allowing each layer to decide how much
gradient information to use. Edge-relevant layers learn small θ (gradient mode).

Architecture:
  - Multi-directional gradient input (4 orient channels + RGB = 7ch input)
  - CDC blocks at 3 scales (16→32→48 channels, ~65K total params)
  - Multi-scale deep supervision (side outputs from each scale)
  - Thin Edge Head trained with skeletonized GT
  - Focal Tversky + Dice loss for clean thin edges

Usage:
  python pidinet_sem.py --mode train --data_dir ./data --use_skeleton_gt
  python pidinet_sem.py --mode infer --image_path ./test.jpg --thin_output
  python pidinet_sem.py --mode info
"""

from __future__ import print_function

# ============================================================================
# IDE Settings
# ============================================================================
IDE_MODE = "info"               # train | infer | info
IDE_DATA_DIR = "./data"         # Path to data/raw/ + data/mask/
IDE_EPOCHS = 10
IDE_LR = 8e-4
IDE_IMG_SIZE = 352
IDE_BATCH_SIZE = 8
IDE_USE_SKELETON_GT = True      # Train thin head with skeletonized GT
IDE_CHECKPOINT = "./checkpoints/best_model.pth"
IDE_RESULT_DIR = "./results"
IDE_IMAGE_PATH = ""
IDE_IMAGE_DIR = ""

import os, sys, time, random, math, argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import cv2, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

@dataclass
class PIDIConfig:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"

    epochs: int = 10
    batch_size: int = 8
    lr: float = 8e-4
    wd: float = 2e-4
    img_size: int = 352
    num_workers: int = 4
    seed: int = 1021
    fp16: bool = False
    grad_clip: float = 0.0

    # Model
    use_directional_input: bool = True   # 4 orient gradient channels
    use_thin_head: bool = True
    cdc_init_theta: float = 0.5          # Initial θ for CDC (0.5 = balanced)

    # Loss
    loss_tversky_alpha: float = 0.7
    loss_tversky_beta: float = 0.3
    loss_tversky_gamma: float = 0.75
    loss_edge_weight: float = 1.0
    loss_thin_weight: float = 1.0

    # Augmentation
    aug_rotation: float = 15.0
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_crop_prob: float = 0.4
    aug_crop_min_size: int = 256
    aug_edge_boost: float = 0.2

    # Training
    val_split: float = 0.2
    lr_scheduler: str = "step"
    lr_milestones: tuple = (4,)
    lr_gamma: float = 0.1
    lr_min: float = 1e-6
    save_interval: int = 1
    val_interval: int = 1
    log_interval: int = 20
    early_stopping_patience: int = 0
    resume_from: str = ""
    pretrained_backbone: str = ""

    # Inference
    thin_output: bool = True
    predict_all_outputs: bool = False

    # Mean pixels (BGR)
    mean_pixels: tuple = (103.939, 116.779, 123.68)


# ============================================================================
# 2. ACTIVATION
# ============================================================================

def smish(input: torch.Tensor) -> torch.Tensor:
    return input * torch.tanh(torch.log(1.0 + torch.sigmoid(input)))


class Smish(nn.Module):
    def forward(self, x): return smish(x)


# ============================================================================
# 3. CENTRAL DIFFERENCE CONVOLUTION (CDC)
# ============================================================================

class CDC(nn.Module):
    """
    Central Difference Convolution.

    y = θ · (x ∗ k_vanilla) + (1-θ) · (x ∗ k_difference) + bias

    where k_difference is k_vanilla with each kernel forced to be zero-mean,
    making it compute weighted differences from center pixel.

    θ is learned per-layer. θ → 1 = vanilla conv, θ → 0 = pure difference conv.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = 1

        # Shared weight for both vanilla and difference paths
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.register_parameter('bias', None)

        # θ: balance between vanilla (1) and difference (0) convolution
        self.theta = nn.Parameter(torch.tensor(0.5))

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vanilla convolution
        out_v = F.conv2d(x, self.weight, self.bias,
                         self.stride, self.padding, self.dilation)

        # Difference convolution: force kernel to be zero-mean
        mean_w = self.weight.mean(dim=(2, 3), keepdim=True)
        zero_mean_w = self.weight - mean_w
        out_d = F.conv2d(x, zero_mean_w, None,
                         self.stride, self.padding, self.dilation)

        # Learnable combination
        theta = torch.sigmoid(self.theta)
        out = theta * out_v + (1 - theta) * out_d

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


class CDCDoubleBlock(nn.Module):
    """Two CDC layers with Smish activation."""
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int = None,
                 stride: int = 1):
        super().__init__()
        if out_ch is None:
            out_ch = mid_ch
        self.conv1 = CDC(in_ch, mid_ch, stride=stride)
        self.conv2 = CDC(mid_ch, out_ch)
        self.act = Smish()

    def forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


# ============================================================================
# 4. DIRECTIONAL GRADIENT FEATURES
# ============================================================================

class DirectionalGradientExtractor(nn.Module):
    """
    Extract 4 directional gradient channels from RGB input.
    Orientations: 0°, 45°, 90°, 135°
    """

    def __init__(self):
        super().__init__()
        # Prewitt-like directional kernels
        k0 = torch.tensor([[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]])   # 0°
        k45 = torch.tensor([[[0., 1., 1.], [-1., 0., 1.], [-1., -1., 0.]]])  # 45°
        k90 = torch.tensor([[[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]])  # 90°
        k135 = torch.tensor([[[1., 1., 0.], [1., 0., -1.], [0., -1., -1.]]]) # 135°

        # Register as buffer (non-trainable)
        kernel = torch.stack([k0, k45, k90, k135])  # (4, 1, 3, 3)
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) RGB image (normalized)
        Returns: (B, 4, H, W) directional gradient magnitudes
        """
        # Convert to grayscale for gradient computation
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        B, C, H, W = gray.shape
        kernel = self.kernel.repeat(1, C, 1, 1)  # (4, 1, 3, 3)

        # Apply directional filters
        grad = F.conv2d(gray, kernel, padding=1)  # (B, 4, H, W)

        # Normalize per channel
        grad_abs = torch.abs(grad)
        grad_max = grad_abs.amax(dim=(2, 3), keepdim=True) + 1e-8
        grad = grad_abs / grad_max

        return grad


# ============================================================================
# 5. PIDINET-SEM MODEL
# ============================================================================

class UpSampleBlock(nn.Module):
    """Upsample + refine for side outputs."""

    def __init__(self, in_ch: int, up_scale: int):
        super().__init__()
        layers = []
        for i in range(up_scale):
            out_ch = 1 if i == up_scale - 1 else 16
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(out_ch, out_ch, 2, stride=2, padding=0))
            in_ch = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ThinEdgeHead(nn.Module):
    """Predicts single-pixel skeleton edges from fused features + coarse edge."""

    def __init__(self, fused_ch: int = 3):
        super().__init__()
        self.conv1 = CDC(fused_ch + 1, 32)
        self.conv2 = CDC(32, 16)
        self.conv3 = nn.Conv2d(16, 1, 1)
        self.act = Smish()

    def forward(self, fused_features, coarse_edge_prob):
        x = torch.cat([fused_features, coarse_edge_prob], dim=1)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


class PiDiNetSEM(nn.Module):
    """
    PiDiNet-inspired edge detector for SEM metrology.

    Uses CDC (Central Difference Convolution) throughout, plus
    directional gradient input for enhanced edge localization.
    """

    def __init__(self, cfg: PIDIConfig = None):
        super().__init__()
        if cfg is None:
            cfg = PIDIConfig()

        in_ch = 3 + (4 if cfg.use_directional_input else 0)
        self.use_directional_input = cfg.use_directional_input
        self.use_thin_head = cfg.use_thin_head

        # Gradient extractor
        if self.use_directional_input:
            self.grad_extractor = DirectionalGradientExtractor()

        # ── Encoder ──────────────────────────────────────────────
        # Scale 1: H/2
        self.stem = CDCDoubleBlock(in_ch, 16, stride=2)

        # Scale 2: H/4
        self.block1 = CDCDoubleBlock(16, 32)
        self.pool1 = nn.MaxPool2d(2)

        # Scale 3: H/8
        self.block2 = CDCDoubleBlock(32, 48)
        self.pool2 = nn.MaxPool2d(2)

        # Deepest
        self.block3 = CDCDoubleBlock(48, 64)

        # ── Side outputs (upsample to input resolution) ─────────
        self.up1 = UpSampleBlock(16, 1)   # stem output: 2x up
        self.up2 = UpSampleBlock(32, 2)   # block1 output: 4x up
        self.up3 = UpSampleBlock(48, 3)   # block2 output: 8x up
        # block3: 8x up (same as block2, need 8x)
        self.up4 = UpSampleBlock(64, 3)

        # ── Fusion ───────────────────────────────────────────────
        self.fusion = nn.Sequential(
            CDC(4, 16), Smish(),
            nn.Conv2d(16, 1, 1)
        )

        # ── Thin head ────────────────────────────────────────────
        if self.use_thin_head:
            self.thin_head = ThinEdgeHead(fused_ch=4)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not isinstance(m, CDC):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _resize_to(self, tensor, target):
        if tensor.shape[2:] != target.shape[2:]:
            return F.interpolate(tensor, size=target.shape[2:],
                                 mode='bilinear', align_corners=False)
        return tensor

    def forward(self, x: torch.Tensor, return_all: bool = False) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        original_shape = (H, W)

        # Pad to multiple of 8
        if H % 8 != 0 or W % 8 != 0:
            pad_h = ((H // 8) + 1) * 8
            pad_w = ((W // 8) + 1) * 8
            x = F.interpolate(x, size=(pad_h, pad_w), mode='bilinear', align_corners=False)

        # Add directional gradient channels
        if self.use_directional_input:
            grad = self.grad_extractor(x)
            x = torch.cat([x, grad], dim=1)

        # ── Encoder forward ──────────────────────────────────
        s1 = self.stem(x)                                          # (B,16,H/2,W/2)
        s2 = self.block1(s1)                                       # (B,32,H/2,W/2)
        s2p = self.pool1(s2)                                       # (B,32,H/4,W/4)
        s3 = self.block2(s2p)                                      # (B,48,H/4,W/4)
        s3p = self.pool2(s3)                                       # (B,48,H/8,W/8)
        s4 = self.block3(s3p)                                      # (B,64,H/8,W/8)

        # ── Side outputs ─────────────────────────────────────
        side1 = self.up1(s1)
        side2 = self.up2(s2)
        side3 = self.up3(s3)
        side4 = self.up4(s4)

        # Align all to same size
        side2 = self._resize_to(side2, side1)
        side3 = self._resize_to(side3, side1)
        side4 = self._resize_to(side4, side1)

        # ── Fusion ────────────────────────────────────────────
        fused_cat = torch.cat([side1, side2, side3, side4], dim=1)
        edge = self.fusion(fused_cat)

        # Crop to original size
        edge = self._resize_to(edge, torch.empty(B, 1, *original_shape, device=x.device))

        outputs = {'edge': edge}

        if return_all:
            outputs['side1'] = self._resize_to(side1, edge)
            outputs['side2'] = self._resize_to(side2, edge)
            outputs['side3'] = self._resize_to(side3, edge)
            outputs['side4'] = self._resize_to(side4, edge)

        # ── Thin head ─────────────────────────────────────────
        if self.use_thin_head:
            fused_resized = self._resize_to(fused_cat, edge)
            edge_prob = torch.sigmoid(edge)
            thin = self.thin_head(fused_resized, edge_prob)
            outputs['thin'] = self._resize_to(thin, edge)

        return outputs


# ============================================================================
# 6. LOSS FUNCTIONS
# ============================================================================

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-8):
    pred = torch.sigmoid(pred)
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.pow(1 - tversky, gamma).mean()


def dice_loss(pred, target, smooth=1e-8):
    pred = torch.sigmoid(pred)
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return (1 - (2 * intersection + smooth) / (union + smooth)).mean()


def bdcn_loss_fn(pred, target, l_weight=1.1):
    target = target.long()
    mask = target.float()
    n_pos = (mask > 0.).sum().float()
    n_neg = (mask <= 0.).sum().float()
    mask[mask > 0.] = n_neg / (n_pos + n_neg)
    mask[mask <= 0.] = 1.1 * n_pos / (n_pos + n_neg)
    pred = torch.sigmoid(pred)
    cost = F.binary_cross_entropy(pred, target.float(), weight=mask.detach(), reduction='none')
    return cost.float().mean((1, 2, 3)).sum()


def combined_loss(outputs, target, skeleton_target=None, cfg=None):
    if cfg is None:
        cfg = PIDIConfig()

    total = cfg.loss_edge_weight * focal_tversky_loss(
        outputs['edge'], target, cfg.loss_tversky_alpha,
        cfg.loss_tversky_beta, cfg.loss_tversky_gamma)

    loss_dict = {'edge': total.item()}

    if 'thin' in outputs and skeleton_target is not None:
        loss_thin = dice_loss(outputs['thin'], skeleton_target)
        loss_dict['thin'] = loss_thin.item()
        total = total + cfg.loss_thin_weight * loss_thin

    # BDCN loss for side outputs
    l_weights = [1.1, 0.7, 1.1, 1.3]
    for i, key in enumerate(['side1', 'side2', 'side3', 'side4']):
        if key in outputs:
            ls = 0.3 * bdcn_loss_fn(outputs[key], target, l_weights[i])
            loss_dict[key] = ls.item()
            total = total + ls

    return total, loss_dict


# ============================================================================
# 7. SKELETONIZATION (Zhang-Suen)
# ============================================================================

def zhang_suen(binary):
    """Zhang-Suen thinning. Input: (H,W) float, output: (H,W) uint8 skeleton."""
    img = (binary > 0.1).astype(np.uint8)
    h, w = img.shape
    skel = img.copy()

    def _nbrs(y, x):
        n = []
        for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)]:
            ny, nx = y + dy, x + dx
            n.append(skel[ny, nx] if 0 <= ny < h and 0 <= nx < w else 0)
        return n

    def _trans(n):
        return sum(1 for i in range(8) if n[i] == 0 and n[(i + 1) % 8] == 1)

    changed = True
    while changed:
        changed = False
        for step in [1, 2]:
            rm = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if skel[y, x] == 0:
                        continue
                    n = _nbrs(y, x)
                    s = sum(n)
                    if not (2 <= s <= 6):
                        continue
                    if _trans(n) != 1:
                        continue
                    if step == 1 and (n[0] * n[2] * n[4] != 0 or n[2] * n[4] * n[6] != 0):
                        continue
                    if step == 2 and (n[0] * n[2] * n[6] != 0 or n[0] * n[4] * n[6] != 0):
                        continue
                    rm.append((y, x))
            for y, x in rm:
                skel[y, x] = 0
                changed = True
    return skel


# ============================================================================
# 8. DATASET
# ============================================================================

class PIDIDataset(Dataset):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, data_dir, cfg, train_mode=True, use_skeleton_gt=False):
        self.cfg = cfg
        self.train_mode = train_mode
        self.use_skeleton_gt = use_skeleton_gt

        raw_dir = os.path.join(data_dir, 'raw')
        mask_dir = os.path.join(data_dir, 'mask')
        if not os.path.isdir(raw_dir) or not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Need {raw_dir} and {mask_dir}")

        raw_files = sorted([f for f in os.listdir(raw_dir)
                           if os.path.splitext(f)[1].lower() in self.IMG_EXTS])
        mask_map = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)
                    if os.path.splitext(f)[1].lower() in self.IMG_EXTS}

        self.samples = []
        for rf in raw_files:
            stem = os.path.splitext(rf)[0]
            if stem in mask_map:
                self.samples.append((os.path.join(raw_dir, rf),
                                     os.path.join(mask_dir, mask_map[stem]), stem))
        print(f"[PIDI Dataset] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, mask_p, stem = self.samples[idx]
        img = cv2.imread(img_p, cv2.IMREAD_COLOR).astype(np.float32)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if mask.max() > 1:
            mask /= 255.

        if self.train_mode:
            img, mask = self._augment(img, mask)

        img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
        mask = cv2.resize(mask, (self.cfg.img_size, self.cfg.img_size),
                          interpolation=cv2.INTER_NEAREST)

        img -= np.array(self.cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)

        result = {
            'images': torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            'labels': torch.from_numpy(mask[np.newaxis, ...].copy()).float(),
            'file_name': stem + '.png'
        }

        if self.use_skeleton_gt:
            skel = zhang_suen(mask)
            result['skeleton'] = torch.from_numpy(skel[np.newaxis, ...]).float()

        return result

    def _augment(self, img, mask):
        c = self.cfg
        h, w = mask.shape
        if c.aug_crop_prob > 0 and random.random() < c.aug_crop_prob:
            sz = min(c.aug_crop_min_size, h, w)
            ch, cw = random.randint(sz, h), random.randint(sz, w)
            t, l = random.randint(0, h - ch), random.randint(0, w - cw)
            img, mask = img[t:t + ch, l:l + cw], mask[t:t + ch, l:l + cw]
        if c.aug_rotation > 0 and random.random() < 0.5:
            a = random.uniform(-c.aug_rotation, c.aug_rotation)
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), a, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                  flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        if c.aug_hflip and random.random() < 0.5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
        if c.aug_vflip and random.random() < 0.5:
            img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)
        if c.aug_brightness > 0:
            img += random.uniform(-c.aug_brightness * 255, c.aug_brightness * 255)
        if c.aug_contrast > 0:
            a = 1 + random.uniform(-c.aug_contrast, c.aug_contrast)
            img = (img - img.mean()) * a + img.mean()
        if c.aug_edge_boost > 0:
            mask[mask > 0.1] += c.aug_edge_boost
            mask = np.clip(mask, 0, 1)
        return img, mask


# ============================================================================
# 9. TRAINING
# ============================================================================

def train(cfg):
    print("=" * 60)
    print("PiDiNet-SEM Training")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_skel = cfg.use_thin_head
    full_ds = PIDIDataset(cfg.data_dir, cfg, train_mode=True, use_skeleton_gt=use_skel)
    n_val = max(1, int(len(full_ds) * cfg.val_split))
    n_train = len(full_ds) - n_val
    train_idx, val_idx = random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed))

    class SubDS:
        def __init__(self, ds, indices, train_mode, use_skel):
            self.ds, self.indices = ds, indices
            self.tm, self.us = train_mode, use_skel
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            om = self.ds.train_mode
            self.ds.train_mode = self.tm
            self.ds.use_skeleton_gt = self.us
            item = self.ds[self.indices[i]]
            self.ds.train_mode = om
            return item

    train_ds = SubDS(full_ds, train_idx.indices, True, use_skel)
    val_ds = SubDS(full_ds, val_idx.indices, False, use_skel)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_ldr = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ldr = DataLoader(val_ds, 1, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    model = PiDiNetSEM(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load pretrained
    if cfg.pretrained_backbone and os.path.isfile(cfg.pretrained_backbone):
        print(f"Loading: {cfg.pretrained_backbone}")
        ckpt = torch.load(cfg.pretrained_backbone, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.lr_scheduler == "step":
        sch = optim.lr_scheduler.MultiStepLR(
            opt, [m * len(train_ldr) for m in cfg.lr_milestones], cfg.lr_gamma)
    elif cfg.lr_scheduler == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg.epochs * len(train_ldr), eta_min=cfg.lr_min)
    else:
        sch = optim.lr_scheduler.StepLR(opt, len(train_ldr), cfg.lr_gamma)

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.fp16)
    best_val = float('inf')
    start = 0

    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val_loss', float('inf'))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(start, cfg.epochs):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0
        for bid, batch in enumerate(train_ldr):
            imgs = batch['images'].to(device)
            lbls = batch['labels'].to(device)
            skels = batch.get('skeleton')
            if skels is not None:
                skels = skels.to(device)

            with torch.amp.autocast('cuda', enabled=cfg.fp16):
                outs = model(imgs, return_all=True)
                loss, ld = combined_loss(outs, lbls, skels, cfg)

            opt.zero_grad()
            if cfg.fp16:
                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

            if cfg.lr_scheduler in ("step", "cosine"):
                sch.step()

            train_loss += loss.item()
            if bid % cfg.log_interval == 0:
                cstr = ' | '.join(f'{k}:{v:.4f}' for k, v in ld.items())
                print(f"  [E{epoch:3d} B{bid:4d}/{len(train_ldr):4d}] "
                      f"L:{loss.item():.4f} ({cstr}) LR:{opt.param_groups[0]['lr']:.2e}")

        train_loss /= len(train_ldr)
        print(f"  Train Loss: {train_loss:.4f}")

        # ── Val ────────────────────────────────────────────
        if cfg.val_interval > 0 and (epoch + 1) % cfg.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for bid, batch in enumerate(val_ldr):
                    imgs = batch['images'].to(device)
                    lbls = batch['labels'].to(device)
                    skels = batch.get('skeleton')
                    if skels is not None:
                        skels = skels.to(device)
                    outs = model(imgs, return_all=True)
                    loss, _ = combined_loss(outs, lbls, skels, cfg)
                    val_loss += loss.item()
            val_loss /= len(val_ldr)
            print(f"  Val Loss: {val_loss:.4f}")

            if cfg.lr_scheduler == "plateau":
                sch.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_val_loss': best_val, 'config': cfg,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))
                print(f"  >>> Best model saved")

        if (epoch + 1) % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_val_loss': best_val, 'config': cfg,
            }, os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth'))

    torch.save({'epoch': cfg.epochs, 'model_state_dict': model.state_dict(),
                'config': cfg},
               os.path.join(cfg.checkpoint_dir, 'final_model.pth'))
    print(f"\nDone. Best val: {best_val:.4f}")


# ============================================================================
# 10. INFERENCE
# ============================================================================

def load(checkpoint_path, device, cfg=None):
    if cfg is None:
        cfg = PIDIConfig()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'config' in ckpt:
        cfg = ckpt['config']
    model = PiDiNetSEM(cfg).to(device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def save_out(tensor, path, original_shape=None, is_thin=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        if not is_thin:
            tensor = torch.sigmoid(tensor)
        tensor = tensor.cpu().detach().numpy()
    img = np.squeeze(tensor)
    img = np.uint8((img - img.min()) / (img.max() - img.min() + 1e-8) * 255)
    img = cv2.bitwise_not(img)
    if original_shape and img.shape != original_shape:
        img = cv2.resize(img, (original_shape[1], original_shape[0]),
                         interpolation=cv2.INTER_NEAREST if is_thin else cv2.INTER_LINEAR)
    cv2.imwrite(path, img)


def infer_single(model, image_path, output_dir, device, cfg):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    orig = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    img_f = img.astype(np.float32)
    img_f -= np.array(cfg.mean_pixels, dtype=np.float32).reshape(1, 1, 3)
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outs = model(img_t)

    save_out(outs['edge'][0], os.path.join(output_dir, f'{stem}_edge.png'), orig)
    if 'thin' in outs and cfg.thin_output:
        save_out(outs['thin'][0], os.path.join(output_dir, 'thin', f'{stem}_thin.png'),
                 orig, is_thin=True)
    print(f"Saved: {output_dir}/")


def infer_dir(model, image_dir, output_dir, device, cfg):
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in exts])
    print(f"Processing {len(files)} images...")
    for f in files:
        infer_single(model, os.path.join(image_dir, f), output_dir, device, cfg)


# ============================================================================
# 11. MODEL INFO
# ============================================================================

def info(cfg):
    print("=" * 60)
    print("PiDiNet-SEM")
    print("=" * 60)
    model = PiDiNetSEM(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n:,}")
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    model.eval()
    with torch.no_grad():
        outs = model(x)
    print(f"Input: (1, 3, {cfg.img_size}, {cfg.img_size})")
    print(f"  Directional input: {'ON' if cfg.use_directional_input else 'OFF'} (7ch)")
    for k, v in outs.items():
        print(f"  {k}: {list(v.shape)}")
    print(f"  Thin Head: {'ON' if cfg.use_thin_head else 'OFF'}")
    print(f"\n  CDC θ values:")
    for name, m in model.named_modules():
        if isinstance(m, CDC):
            print(f"    {name}: θ={torch.sigmoid(m.theta).item():.3f}")


# ============================================================================
# 12. CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='PiDiNet-SEM')
    p.add_argument('--mode', required=True, choices=['train', 'infer', 'info'])
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--result_dir', default='./results')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--wd', type=float, default=2e-4)
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--use_skeleton_gt', action='store_true', default=False)
    p.add_argument('--pretrained_backbone', default='')
    p.add_argument('--resume_from', default='')
    p.add_argument('--image_path', default='')
    p.add_argument('--image_dir', default='')
    p.add_argument('--checkpoint', default='./checkpoints/best_model.pth')
    p.add_argument('--thin_output', action='store_true', default=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PIDIConfig()
    for k, v in vars(args).items():
        if k in PIDIConfig.__dataclass_fields__:
            setattr(cfg, k, v)

    if args.mode == 'info':
        info(cfg)
    elif args.mode == 'train':
        train(cfg)
    elif args.mode == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load(args.checkpoint, device, cfg)
        if args.image_path:
            infer_single(model, args.image_path, cfg.result_dir, device, cfg)
        elif args.image_dir:
            infer_dir(model, args.image_dir, cfg.result_dir, device, cfg)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.argv.extend(['--mode', IDE_MODE])
        if IDE_MODE == 'train':
            sys.argv.extend(['--data_dir', IDE_DATA_DIR])
            sys.argv.extend(['--epochs', str(IDE_EPOCHS), '--lr', str(IDE_LR),
                            '--img_size', str(IDE_IMG_SIZE), '--batch_size', str(IDE_BATCH_SIZE)])
            if IDE_USE_SKELETON_GT:
                sys.argv.append('--use_skeleton_gt')
        elif IDE_MODE == 'infer':
            sys.argv.extend(['--checkpoint', IDE_CHECKPOINT, '--result_dir', IDE_RESULT_DIR])
            if IDE_IMAGE_PATH:
                sys.argv.extend(['--image_path', IDE_IMAGE_PATH])
            if IDE_IMAGE_DIR:
                sys.argv.extend(['--image_dir', IDE_IMAGE_DIR])
    main()
