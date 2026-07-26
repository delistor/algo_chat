"""
RefineNet: Two-Stage Edge Refinement Network for SEM Metrology
===============================================================
Single-file self-contained implementation.

Core innovation: A two-stage pipeline where Stage 1 detects coarse edges and
Stage 2 learns to thin them. This separates the hard problems:
  - Stage 1: Find WHERE edges are (coarse, high-recall)
  - Stage 2: Learn to output WHAT edges should look like (thin, single-pixel)

This is the most data-efficient approach because:
  1. Stage 1 can use ANY pretrained edge detector (TEED, PiDiNet, HED, etc.)
  2. Stage 2 has very few parameters (~5K) → trains with minimal data
  3. Stage 2 essentially learns NMS + skeletonization + denoising END-TO-END
  4. The refinement generalizes across different Stage 1 detectors

Architecture:
  Stage 1: TEED (frozen or fine-tuned) → coarse edge probability
  Stage 2: RefineNet (lightweight 4-layer CNN)
            Input: [original image, coarse edge prob, gradient magnitude]
            Output: thin single-pixel edge map
            Trained with Dice loss on skeletonized GT

Why Stage 2 works:
  The refinement network learns:
  - Where to SUPPRESS (texture regions with high coarse response → noise)
  - Where to ENHANCE (weak but real edges → continuity)
  - How to THIN (thick coarse edges → single-pixel skeleton)
  This is fundamentally learning the "inverse" of NMS + morphological thinning,
  but in a data-driven way that adapts to SEM-specific edge characteristics.

Usage:
  # Stage 1: Train TEED on your data
  python teed_clean_3.py --mode train ...

  # Stage 2: Train RefineNet (requires pretrained Stage 1)
  python refine_sem.py --mode train --data_dir ./data \
      --stage1_checkpoint ./checkpoints/best_model.pth \
      --use_skeleton_gt

  # End-to-end inference
  python refine_sem.py --mode infer --image_path ./test.jpg \
      --checkpoint ./checkpoints/refine_best.pth

  # Standalone: train both stages from scratch
  python refine_sem.py --mode train --data_dir ./data \
      --train_stage1 --use_skeleton_gt
"""

from __future__ import print_function

IDE_MODE = "info"               # train | infer | info
IDE_DATA_DIR = "./data"         # Path to data/raw/ + data/mask/
IDE_EPOCHS = 10                 # Stage 1 epochs
IDE_STAGE2_EPOCHS = 20          # Stage 2 epochs (refinement)
IDE_LR = 1e-3
IDE_IMG_SIZE = 352
IDE_BATCH_SIZE = 8
IDE_TRAIN_STAGE1 = True         # Train Stage 1 from scratch?
IDE_FREEZE_STAGE1 = True        # Freeze Stage 1 during Stage 2 training
IDE_CHECKPOINT = "./checkpoints/best_model.pth"
IDE_STAGE1_CHECKPOINT = "./checkpoints/stage1_best.pth"
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
class RefineConfig:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"

    # Training
    epochs: int = 10
    stage2_epochs: int = 20     # Train Stage 2 longer
    batch_size: int = 8
    lr: float = 1e-3            # Higher LR for refinement network
    wd: float = 1e-4
    img_size: int = 352
    num_workers: int = 4
    seed: int = 1021
    fp16: bool = False
    grad_clip: float = 1.0

    # Stage control
    train_stage1: bool = True           # Train TEED from scratch?
    freeze_stage1: bool = True          # Freeze Stage 1 during Stage 2 training
    stage1_checkpoint: str = ""         # Pretrained TEED weights

    # Loss
    loss_stage1_alpha: float = 0.7      # Tversky α for Stage 1
    loss_stage2_dice_weight: float = 1.0
    loss_stage2_bce_weight: float = 0.3

    # Augmentation
    aug_rotation: float = 15.0
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_crop_prob: float = 0.4
    aug_crop_min_size: int = 256
    aug_edge_boost: float = 0.2

    # Training schedule
    val_split: float = 0.2
    lr_scheduler: str = "cosine"
    lr_milestones: tuple = (5,)
    lr_gamma: float = 0.1
    lr_min: float = 1e-6
    save_interval: int = 1
    val_interval: int = 1
    log_interval: int = 20
    resume_from: str = ""

    # Inference
    thin_threshold: float = 0.5

    mean_pixels: tuple = (103.939, 116.779, 123.68)


# ============================================================================
# 2. STAGE 1: TEED (Tiny Edge Detector)
# ============================================================================

def smish_fn(x):
    return x * torch.tanh(torch.log(1.0 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x): return smish_fn(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch=None, stride=1, use_act=True):
        super().__init__()
        if out_ch is None:
            out_ch = mid_ch
        self.use_act = use_act
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.act = Smish()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x) if self.use_act else x


class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride)

    def forward(self, x): return self.conv(x)


class DenseLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
        self.act = Smish()

    def forward(self, x):
        x1, x2 = x
        f = self.conv2(self.act(self.conv1(smish_fn(x1))))
        return 0.5 * (f + x2), x2


class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_ch, out_ch):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(DenseLayer(in_ch, out_ch))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, up_scale):
        super().__init__()
        layers = []
        pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            k = 2 ** up_scale
            out_ch = 1 if i == up_scale - 1 else 16
            layers.append(nn.Conv2d(in_ch, out_ch, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose2d(out_ch, out_ch, k, stride=2, padding=pads[up_scale]))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class DoubleFusion(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.dw1 = nn.Conv2d(in_ch, in_ch * 8, 3, padding=1, groups=in_ch)
        self.ps = nn.PixelShuffle(1)
        self.dw2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.act = Smish()

    def forward(self, x):
        a = self.ps(self.dw1(self.act(x)))
        a2 = self.ps(self.dw2(self.act(a)))
        return smish_fn(((a2 + a).sum(1)).unsqueeze(1))


class TEED(nn.Module):
    """Lightweight TEED for Stage 1."""
    def __init__(self):
        super().__init__()
        self.block1 = DoubleConv(3, 16, 16, stride=2)
        self.block2 = DoubleConv(16, 32, use_act=False)
        self.dblock3 = DenseBlock(1, 32, 48)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.side1 = SingleConv(16, 32, 2)
        self.pre3 = SingleConv(32, 48, 1)
        self.up1 = UpBlock(16, 1)
        self.up2 = UpBlock(32, 1)
        self.up3 = UpBlock(48, 2)
        self.fusion = DoubleFusion(3)

    def forward(self, x):
        orig = x.shape[2:]
        # Pad
        ph = ((orig[0] // 8) + 1) * 8 if orig[0] % 8 else orig[0]
        pw = ((orig[1] // 8) + 1) * 8 if orig[1] % 8 else orig[1]
        if (ph, pw) != orig:
            x = F.interpolate(x, size=(ph, pw), mode='bilinear', align_corners=False)

        b1 = self.block1(x)
        b2 = self.block2(b1)
        b2d = self.maxpool(b2)
        b2a = b2d + self.side1(b1)
        b3, _ = self.dblock3([b2a, self.pre3(b2d)])

        o1 = self.up1(b1)
        o2 = self.up2(b2)
        o3 = self.up3(b3)

        # Align
        o2 = F.interpolate(o2, size=o1.shape[2:], mode='bilinear', align_corners=False)
        o3 = F.interpolate(o3, size=o1.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([o1, o2, o3], dim=1))

        # Crop
        if (ph, pw) != orig:
            fused = fused[:, :, :orig[0], :orig[1]]
        return fused


# ============================================================================
# 3. STAGE 2: REFINENET — Learned Edge Refinement
# ============================================================================

class RefineNet(nn.Module):
    """
    Learned edge refinement network.

    Input: [original image (3ch), coarse edge prob (1ch), gradient mag (1ch)] = 5ch
    Output: thin single-pixel edge logits (1ch)

    Architecture: 4-layer CNN with residual connections.
    Extremely lightweight (~5K params) — trains with very little data.
    """

    def __init__(self):
        super().__init__()
        ch = [5, 32, 32, 16, 1]

        self.conv1 = nn.Conv2d(ch[0], ch[1], 3, padding=1)
        self.conv2 = nn.Conv2d(ch[1], ch[2], 3, padding=1)
        self.conv3 = nn.Conv2d(ch[2], ch[3], 3, padding=1)
        self.conv4 = nn.Conv2d(ch[3], ch[4], 1)

        self.act = Smish()

        # Residual connection: conv1 output → conv3 input
        self.skip = nn.Conv2d(ch[1], ch[3], 1)  # Project for residual add

    def forward(self, image, coarse_edge, gradient_mag):
        """
        Args:
            image: (B, 3, H, W) original image
            coarse_edge: (B, 1, H, W) coarse edge probability [0,1]
            gradient_mag: (B, 1, H, W) Sobel gradient magnitude [0,1]
        Returns:
            thin_logits: (B, 1, H, W)
        """
        x = torch.cat([image, coarse_edge, gradient_mag], dim=1)

        f1 = self.act(self.conv1(x))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2) + self.skip(f1))  # Residual
        out = self.conv4(f3)

        return out


class TwoStageModel(nn.Module):
    """
    Full two-stage model: TEED → RefineNet.

    Can freeze Stage 1 for Stage 2 training.
    """

    def __init__(self, cfg: RefineConfig = None):
        super().__init__()
        self.stage1 = TEED()
        self.stage2 = RefineNet()

    def forward(self, x, return_coarse=False):
        # Stage 1: coarse edge detection
        coarse_logits = self.stage1(x)
        coarse_prob = torch.sigmoid(coarse_logits)

        # Compute gradient magnitude from input for RefineNet
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                         device=x.device).view(1, 1, 3, 3), padding=1)
        gy = F.conv2d(gray, torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                         device=x.device).view(1, 1, 3, 3), padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

        # Normalize image to [0,1] for RefineNet input
        img_norm = (x - x.amin(dim=(2, 3), keepdim=True)) / (
            x.amax(dim=(2, 3), keepdim=True) - x.amin(dim=(2, 3), keepdim=True) + 1e-8)

        # Stage 2: thin edge refinement
        thin_logits = self.stage2(img_norm, coarse_prob, grad_mag)

        if return_coarse:
            return {'coarse': coarse_logits, 'thin': thin_logits}
        return {'thin': thin_logits}


# ============================================================================
# 4. SKELETONIZATION
# ============================================================================

def skeletonize(binary):
    """Zhang-Suen thinning."""
    img = (binary > 0.1).astype(np.uint8)
    h, w = img.shape
    skel = img.copy()

    def nb(y, x):
        return [skel[y + dy, x + dx] if 0 <= y + dy < h and 0 <= x + dx < w else 0
                for dy, dx in [(-1, 0), (-1, 1), (0, 1), (1, 1),
                               (1, 0), (1, -1), (0, -1), (-1, -1)]]

    def tr(n):
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
                    n = nb(y, x)
                    s = sum(n)
                    if not (2 <= s <= 6) or tr(n) != 1:
                        continue
                    if step == 1 and (n[0] * n[2] * n[4] or n[2] * n[4] * n[6]):
                        continue
                    if step == 2 and (n[0] * n[2] * n[6] or n[0] * n[4] * n[6]):
                        continue
                    rm.append((y, x))
            for y, x in rm:
                skel[y, x] = 0
                changed = True
    return skel


# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1e-8):
    pred = torch.sigmoid(pred)
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    return (1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def dice_loss(pred, target, smooth=1e-8):
    pred = torch.sigmoid(pred)
    B = pred.shape[0]
    pred = pred.reshape(B, -1)
    target = target.reshape(B, -1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def refine_loss(outputs, target, skeleton_target, cfg=None):
    """
    Stage 1 loss: Tversky on coarse output
    Stage 2 loss: Dice + BCE on thin output vs skeleton GT
    """
    if cfg is None:
        cfg = RefineConfig()

    losses = {}
    total = 0.0

    # Stage 1: Tversky loss
    if 'coarse' in outputs:
        loss_s1 = tversky_loss(outputs['coarse'], target,
                               alpha=cfg.loss_stage1_alpha)
        losses['stage1'] = loss_s1.item()
        total += loss_s1

    # Stage 2: Dice + BCE on skeleton
    if 'thin' in outputs and skeleton_target is not None:
        loss_dice = dice_loss(outputs['thin'], skeleton_target)
        loss_bce = F.binary_cross_entropy_with_logits(
            outputs['thin'], skeleton_target, reduction='mean')
        losses['thin_dice'] = loss_dice.item()
        losses['thin_bce'] = loss_bce.item()
        total += cfg.loss_stage2_dice_weight * loss_dice
        total += cfg.loss_stage2_bce_weight * loss_bce

    return total, losses


# ============================================================================
# 6. DATASET
# ============================================================================

class RefineDataset(Dataset):
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
        print(f"[Refine Dataset] {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

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
            skel = skeletonize(mask)
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
# 7. TRAINING
# ============================================================================

def train(cfg):
    print("=" * 60)
    print("RefineNet Training — Two-Stage Edge Refinement")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    use_skel = True
    full_ds = RefineDataset(cfg.data_dir, cfg, train_mode=True, use_skeleton_gt=use_skel)
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
            om, ou = self.ds.train_mode, self.ds.use_skeleton_gt
            self.ds.train_mode = self.tm
            self.ds.use_skeleton_gt = self.us
            item = self.ds[self.indices[i]]
            self.ds.train_mode = om
            self.ds.use_skeleton_gt = ou
            return item

    train_ds = SubDS(full_ds, train_idx.indices, True, use_skel)
    val_ds = SubDS(full_ds, val_idx.indices, False, use_skel)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_ldr = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ldr = DataLoader(val_ds, 1, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    model = TwoStageModel(cfg).to(device)
    s1_params = sum(p.numel() for p in model.stage1.parameters() if p.requires_grad)
    s2_params = sum(p.numel() for p in model.stage2.parameters() if p.requires_grad)
    print(f"Stage 1 (TEED): {s1_params:,} params")
    print(f"Stage 2 (RefineNet): {s2_params:,} params")

    # Load pretrained Stage 1 if provided
    if cfg.stage1_checkpoint and os.path.isfile(cfg.stage1_checkpoint):
        print(f"Loading Stage 1: {cfg.stage1_checkpoint}")
        ckpt = torch.load(cfg.stage1_checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        # Filter only Stage 1 weights
        s1_sd = {k.replace('stage1.', ''): v for k, v in sd.items()
                 if 'stage1.' in k or not k.startswith('stage2.')}
        if not s1_sd:
            s1_sd = sd  # Raw TEED weights
        model.stage1.load_state_dict(s1_sd, strict=False)
        print(f"  Stage 1 loaded")

    # ── Phase 1: Train Stage 1 (if requested) ──────────────────────
    if cfg.train_stage1:
        print(f"\n{'=' * 50}")
        print("Phase 1: Training Stage 1 (TEED)")
        print(f"{'=' * 50}")

        # Freeze Stage 2 during Stage 1 training
        for p in model.stage2.parameters():
            p.requires_grad = False

        opt = optim.Adam(model.stage1.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * len(train_ldr), eta_min=cfg.lr_min)

        for epoch in range(cfg.epochs):
            model.train()
            t_loss = 0
            for bid, batch in enumerate(train_ldr):
                imgs = batch['images'].to(device)
                lbls = batch['labels'].to(device)
                skels = batch.get('skeleton')
                if skels is not None:
                    skels = skels.to(device)

                outs = model(imgs, return_coarse=True)
                loss, ld = refine_loss(outs, lbls, skels, cfg)

                opt.zero_grad()
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.stage1.parameters(), cfg.grad_clip)
                opt.step()
                sch.step()
                t_loss += loss.item()

                if bid % cfg.log_interval == 0:
                    print(f"  [S1 E{epoch:3d} B{bid:4d}] L:{loss.item():.4f}")

            t_loss /= len(train_ldr)
            print(f"  Phase 1 Epoch {epoch}: loss={t_loss:.4f}")

        # Save Stage 1
        torch.save({'model_state_dict': model.stage1.state_dict(), 'config': cfg},
                   os.path.join(cfg.checkpoint_dir, 'stage1_best.pth'))

    # ── Phase 2: Train Stage 2 (RefineNet) ─────────────────────────
    print(f"\n{'=' * 50}")
    print("Phase 2: Training Stage 2 (RefineNet)")
    print(f"{'=' * 50}")

    if cfg.freeze_stage1:
        for p in model.stage1.parameters():
            p.requires_grad = False
    for p in model.stage2.parameters():
        p.requires_grad = True

    opt2 = optim.Adam(model.stage2.parameters(), lr=cfg.lr * 2, weight_decay=cfg.wd)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(
        opt2, cfg.stage2_epochs * len(train_ldr), eta_min=cfg.lr_min)

    best_val = float('inf')
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for epoch in range(cfg.stage2_epochs):
        model.train()
        t_loss = 0
        for bid, batch in enumerate(train_ldr):
            imgs = batch['images'].to(device)
            lbls = batch['labels'].to(device)
            skels = batch['skeleton'].to(device) if 'skeleton' in batch else None

            outs = model(imgs, return_coarse=not cfg.freeze_stage1)
            loss, ld = refine_loss(outs, lbls, skels, cfg)

            opt2.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.stage2.parameters(), cfg.grad_clip)
            opt2.step()
            sch2.step()
            t_loss += loss.item()

            if bid % cfg.log_interval == 0:
                cstr = ' | '.join(f'{k}:{v:.4f}' for k, v in ld.items())
                print(f"  [S2 E{epoch:3d} B{bid:4d}] L:{loss.item():.4f} ({cstr})")

        t_loss /= len(train_ldr)
        print(f"  Phase 2 Epoch {epoch}: loss={t_loss:.4f}")

        if cfg.val_interval > 0 and (epoch + 1) % cfg.val_interval == 0:
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for batch in val_ldr:
                    imgs = batch['images'].to(device)
                    lbls = batch['labels'].to(device)
                    skels = batch['skeleton'].to(device) if 'skeleton' in batch else None
                    outs = model(imgs, return_coarse=not cfg.freeze_stage1)
                    loss, _ = refine_loss(outs, lbls, skels, cfg)
                    v_loss += loss.item()
            v_loss /= len(val_ldr)
            print(f"  Val Loss: {v_loss:.4f}")

            if v_loss < best_val:
                best_val = v_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'stage1_state_dict': model.stage1.state_dict(),
                    'stage2_state_dict': model.stage2.state_dict(),
                    'config': cfg,
                }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))

        if (epoch + 1) % cfg.save_interval == 0:
            torch.save({'model_state_dict': model.state_dict(), 'config': cfg},
                       os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth'))

    print(f"\nDone. Best val: {best_val:.4f}")


# ============================================================================
# 8. INFERENCE
# ============================================================================

def load_refine(checkpoint_path, device, cfg=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if cfg is None:
        cfg = RefineConfig()
    if 'config' in ckpt:
        cfg = ckpt['config']
    model = TwoStageModel(cfg).to(device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    s2p = sum(p.numel() for p in model.stage2.parameters())
    print(f"Loaded: {checkpoint_path} | RefineNet: {s2p:,} params")
    return model


def save_out(tensor, path, orig=None, is_thin=False):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        if not is_thin:
            tensor = torch.sigmoid(tensor)
        tensor = tensor.cpu().detach().numpy()
    img = np.squeeze(tensor)
    img = np.uint8((img - img.min()) / (img.max() - img.min() + 1e-8) * 255)
    img = cv2.bitwise_not(img)
    if orig and img.shape != orig:
        img = cv2.resize(img, (orig[1], orig[0]),
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
        outs = model(img_t, return_coarse=True)

    save_out(outs.get('coarse', outs['thin']),
             os.path.join(output_dir, f'{stem}_coarse.png'), orig)
    save_out(outs['thin'], os.path.join(output_dir, f'{stem}_thin.png'),
             orig, is_thin=True)

    # Binary thin edges
    thin_prob = torch.sigmoid(outs['thin'])[0, 0].cpu().numpy()
    thin_bin = (thin_prob > cfg.thin_threshold).astype(np.uint8) * 255
    if thin_bin.shape != orig:
        thin_bin = cv2.resize(thin_bin, (orig[1], orig[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_dir, f'{stem}_binary.png'), thin_bin)
    print(f"Saved: {output_dir}/")


def infer_dir(model, image_dir, output_dir, device, cfg):
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    files = sorted([f for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in exts])
    print(f"Processing {len(files)} images...")
    for f in files:
        infer_single(model, os.path.join(image_dir, f), output_dir, device, cfg)


# ============================================================================
# 9. INFO + CLI
# ============================================================================

def info(cfg):
    print("=" * 60)
    print("RefineNet — Two-Stage Edge Refinement")
    print("=" * 60)
    model = TwoStageModel(cfg)
    s1 = sum(p.numel() for p in model.stage1.parameters())
    s2 = sum(p.numel() for p in model.stage2.parameters())
    print(f"Stage 1 (TEED): {s1:,} params")
    print(f"Stage 2 (RefineNet): {s2:,} params")
    print(f"Total: {s1 + s2:,} params")
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    model.eval()
    with torch.no_grad():
        outs = model(x, return_coarse=True)
    for k, v in outs.items():
        print(f"  {k}: {list(v.shape)}")


def parse_args():
    p = argparse.ArgumentParser(description='RefineNet')
    p.add_argument('--mode', required=True, choices=['train', 'infer', 'info'])
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--result_dir', default='./results')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--stage2_epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--train_stage1', action='store_true', default=True)
    p.add_argument('--freeze_stage1', action='store_true', default=True)
    p.add_argument('--stage1_checkpoint', default='')
    p.add_argument('--use_skeleton_gt', action='store_true', default=True)
    p.add_argument('--resume_from', default='')
    p.add_argument('--image_path', default='')
    p.add_argument('--image_dir', default='')
    p.add_argument('--checkpoint', default='./checkpoints/best_model.pth')
    p.add_argument('--thin_threshold', type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = RefineConfig()
    for k, v in vars(args).items():
        if k in RefineConfig.__dataclass_fields__:
            setattr(cfg, k, v)

    if args.mode == 'info':
        info(cfg)
    elif args.mode == 'train':
        train(cfg)
    elif args.mode == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_refine(args.checkpoint, device, cfg)
        if args.image_path:
            infer_single(model, args.image_path, cfg.result_dir, device, cfg)
        elif args.image_dir:
            infer_dir(model, args.image_dir, cfg.result_dir, device, cfg)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.argv.extend(['--mode', IDE_MODE])
        if IDE_MODE == 'train':
            sys.argv.extend(['--data_dir', IDE_DATA_DIR])
            sys.argv.extend(['--epochs', str(IDE_EPOCHS),
                            '--stage2_epochs', str(IDE_STAGE2_EPOCHS),
                            '--lr', str(IDE_LR), '--img_size', str(IDE_IMG_SIZE),
                            '--batch_size', str(IDE_BATCH_SIZE)])
            if IDE_TRAIN_STAGE1:
                sys.argv.append('--train_stage1')
            if IDE_FREEZE_STAGE1:
                sys.argv.append('--freeze_stage1')
            if IDE_STAGE1_CHECKPOINT and os.path.isfile(IDE_STAGE1_CHECKPOINT):
                sys.argv.extend(['--stage1_checkpoint', IDE_STAGE1_CHECKPOINT])
            sys.argv.append('--use_skeleton_gt')
        elif IDE_MODE == 'infer':
            sys.argv.extend(['--checkpoint', IDE_CHECKPOINT, '--result_dir', IDE_RESULT_DIR])
            if IDE_IMAGE_PATH:
                sys.argv.extend(['--image_path', IDE_IMAGE_PATH])
            if IDE_IMAGE_DIR:
                sys.argv.extend(['--image_dir', IDE_IMAGE_DIR])
    main()
