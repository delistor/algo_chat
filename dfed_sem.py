"""
DFED: Distance Field Edge Detector for SEM Metrology
=====================================================
Single-file self-contained implementation.

Core innovation: Instead of classifying pixels as edge/non-edge (a hard,
imbalanced problem), DFED regresses a smooth distance field where each pixel
value = distance to the nearest edge (normalized to [0, 1]).

  distance = 0  →  pixel IS on the edge
  distance = 1  →  pixel is FAR from any edge

Edges are extracted as LOCAL MINIMA (valleys) of this field. This naturally:
  1. Produces single-pixel wide edges (at the valley floor)
  2. Is robust to class imbalance (regression vs classification)
  3. Provides rich spatial information for metrology
  4. Handles varying edge thickness gracefully

The distance field also provides edge confidence: deeper valleys = stronger edges.

Architecture:
  - Lightweight U-Net style encoder-decoder (~70K params)
  - Distance field regression head
  - Edge probability derived as: edge_prob = exp(-dist² / 2σ²)
  - Thin edge extraction: find valleys → skeleton

Loss:
  - L1 loss on distance field (weighted: higher weight near edges)
  - Gradient consistency loss (encourages smooth distance gradient)
  - Optional: edge probability BCE for the derived edge map

Edge extraction (inference):
  1. Predict distance field
  2. Find local minima (valleys) → single-pixel skeleton
  3. Or threshold edge_prob = exp(-dist²/2σ²) for probability map

Usage:
  python dfed_sem.py --mode train --data_dir ./data
  python dfed_sem.py --mode infer --image_path ./test.jpg
  python dfed_sem.py --mode info
"""

from __future__ import print_function

IDE_MODE = "info"               # train | infer | info
IDE_DATA_DIR = "./data"         # Path to data/raw/ + data/mask/
IDE_EPOCHS = 10
IDE_LR = 1e-3
IDE_IMG_SIZE = 352
IDE_BATCH_SIZE = 8
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
class DFEDConfig:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"

    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-3
    wd: float = 1e-4
    img_size: int = 352
    num_workers: int = 4
    seed: int = 1021
    fp16: bool = False
    grad_clip: float = 1.0

    # Model
    base_channels: int = 16     # First layer channels
    depth: int = 3              # Encoder depth (3 = 3 maxpool = 8x down)
    dist_sigma: float = 0.05    # σ for edge_prob = exp(-dist²/2σ²)

    # Loss weights
    loss_l1_weight: float = 1.0        # L1 on distance field
    loss_grad_weight: float = 0.1      # Gradient consistency
    loss_edge_weight: float = 0.3      # BCE on derived edge prob

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
    lr_scheduler: str = "cosine"
    lr_milestones: tuple = (5,)
    lr_gamma: float = 0.1
    lr_min: float = 1e-6
    save_interval: int = 1
    val_interval: int = 1
    log_interval: int = 20
    early_stopping_patience: int = 0
    resume_from: str = ""
    pretrained: str = ""

    # Inference
    edge_mode: str = "valley"   # "valley" | "prob" — how to extract edges
    edge_threshold: float = 0.3

    mean_pixels: tuple = (103.939, 116.779, 123.68)


# ============================================================================
# 2. ACTIVATION
# ============================================================================

def smish(x):
    return x * torch.tanh(torch.log(1.0 + torch.sigmoid(x)))


class Smish(nn.Module):
    def forward(self, x): return smish(x)


# ============================================================================
# 3. DISTANCE FIELD MODEL
# ============================================================================

class ConvBlock(nn.Module):
    """Double conv: Conv → Smish → Conv → Smish."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Smish(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Smish(),
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """ConvBlock + MaxPool downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f = self.conv(x)
        return f, self.pool(f)


class DecoderBlock(nn.Module):
    """Upsample + skip connection + ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Align spatial dims
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DFED(nn.Module):
    """
    Distance Field Edge Detector.

    Predicts distance from each pixel to nearest edge.
    Edge = local minimum (valley) of the distance field.
    """

    def __init__(self, cfg: DFEDConfig = None):
        super().__init__()
        if cfg is None:
            cfg = DFEDConfig()
        ch = cfg.base_channels
        d = cfg.depth

        # ── Encoder ──────────────────────────────────────────
        self.enc1 = EncoderBlock(3, ch)         # H/2
        self.enc2 = EncoderBlock(ch, ch * 2)    # H/4
        self.enc3 = EncoderBlock(ch * 2, ch * 4)  # H/8

        # Bottleneck
        self.bottleneck = ConvBlock(ch * 4, ch * 8)

        # ── Decoder ──────────────────────────────────────────
        self.dec3 = DecoderBlock(ch * 8, ch * 4, ch * 4)     # H/4
        self.dec2 = DecoderBlock(ch * 4, ch * 2, ch * 2)     # H/2
        self.dec1 = DecoderBlock(ch * 2, ch, ch)             # H

        # ── Heads ────────────────────────────────────────────
        # Distance field: [0, 1] where 0 = edge
        self.dist_head = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), Smish(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Edge probability: derived from distance field
        # This is a small CNN that can apply non-linear refinement
        self.edge_head = nn.Sequential(
            nn.Conv2d(ch + 1, 32, 3, padding=1), Smish(),
            nn.Conv2d(32, 1, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        orig = (H, W)

        # Pad to multiple of 8
        ph = ((H // 8) + 1) * 8 if H % 8 else H
        pw = ((W // 8) + 1) * 8 if W % 8 else W
        if (ph, pw) != (H, W):
            x = F.interpolate(x, size=(ph, pw), mode='bilinear', align_corners=False)

        # Encoder
        f1, p1 = self.enc1(x)       # f1: (B,ch,H/2,W/2), p1: (B,ch,H/4,W/4)
        f2, p2 = self.enc2(p1)      # f2: (B,ch*2,H/4,W/4), p2: (B,ch*2,H/8,W/8)
        f3, p3 = self.enc3(p2)      # f3: (B,ch*4,H/8,W/8), p3: (B,ch*4,H/16,W/16)

        # Bottleneck
        b = self.bottleneck(p3)     # (B,ch*8,H/16,W/16)

        # Decoder
        d3 = self.dec3(b, f3)       # (B,ch*4,H/8,W/8)
        d2 = self.dec2(d3, f2)      # (B,ch*2,H/4,W/4)
        d1 = self.dec1(d2, f1)      # (B,ch,H/2,W/2)

        # Final up to original size
        d1 = F.interpolate(d1, size=(ph, pw), mode='bilinear', align_corners=False)

        # Distance field
        dist = self.dist_head(d1)  # (B, 1, H, W)

        # Edge probability (combine features + distance field)
        edge_in = torch.cat([d1, dist], dim=1)
        edge_logits = self.edge_head(edge_in)

        # Crop to original
        if (ph, pw) != orig:
            dist = dist[:, :, :H, :W]
            edge_logits = edge_logits[:, :, :H, :W]

        return {'dist': dist, 'edge': edge_logits}


# ============================================================================
# 4. DISTANCE FIELD → EDGE EXTRACTION
# ============================================================================

def dist_to_edge(dist_field, sigma=0.05):
    """Convert distance field to edge probability: exp(-d²/2σ²)."""
    return torch.exp(-dist_field ** 2 / (2 * sigma ** 2))


def find_valleys(dist_field, threshold=0.3):
    """
    Find valleys (local minima) in distance field → single-pixel edges.

    A pixel is a valley if it's smaller than its neighbors in the gradient direction.
    This is like a learned NMS that uses the distance field's geometry.

    Returns binary edge map.
    """
    import numpy as np
    if isinstance(dist_field, torch.Tensor):
        dist_field = dist_field.cpu().numpy()
    dist = np.squeeze(dist_field)
    h, w = dist.shape

    # Find pixels that are local minima in 3x3 neighborhood
    edges = np.zeros((h, w), dtype=np.uint8)
    padded = np.pad(dist, 1, mode='edge')

    for y in range(1, h + 1):
        for x in range(1, w + 1):
            patch = padded[y - 1:y + 2, x - 1:x + 2]
            center = padded[y, x]
            # Valley: center is smaller than all neighbors AND below threshold
            if center < threshold and center <= patch.min():
                edges[y - 1, x - 1] = 1

    return edges


# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

def compute_gt_distance(mask, max_dist=None):
    """
    Compute ground truth distance field from binary edge mask.
    Uses Euclidean distance transform.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.squeeze(mask)
    binary = (mask > 0.1).astype(np.uint8)

    # Distance to nearest edge
    dist = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Normalize
    if max_dist is None:
        max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist
    # Clip to [0, 1]
    dist = np.clip(dist, 0, 1)

    return dist


def dfed_loss(outputs, target, cfg=None):
    """
    Combined DFED loss:
      L1: predict the distance field accurately
      Gradient consistency: edges of distance field should align with GT edges
      Edge BCE: derived edge probability should match GT
    """
    if cfg is None:
        cfg = DFEDConfig()

    pred_dist = outputs['dist']
    edge_logits = outputs['edge']

    B = target.shape[0]
    losses = {}

    # Compute GT distance field for each sample
    total = 0.0
    for i in range(B):
        gt_dist = compute_gt_distance(target[i])
        gt_dist_t = torch.from_numpy(gt_dist).float().to(pred_dist.device)

        # L1 loss on distance field (edge-weighted)
        edge_mask = (gt_dist_t < 0.1).float()  # Near edge = high weight
        weight = 1.0 + 10.0 * edge_mask  # 11x weight near edges, 1x elsewhere
        l1 = (torch.abs(pred_dist[i, 0] - gt_dist_t) * weight).mean()
        total += cfg.loss_l1_weight * l1

        # Gradient consistency: ∇(dist) magnitude should be large at edges
        if cfg.loss_grad_weight > 0:
            gy = torch.abs(pred_dist[i, 0, 1:] - pred_dist[i, 0, :-1])
            gx = torch.abs(pred_dist[i, 0, :, 1:] - pred_dist[i, 0, :, :-1])
            grad_mag = torch.cat([gy.mean().unsqueeze(0), gx.mean().unsqueeze(0)]).mean()
            # Encourage gradient near edges
            grad_loss = F.mse_loss(grad_mag, torch.tensor(0.01, device=grad_mag.device))
            total += cfg.loss_grad_weight * grad_loss

    total = total / B
    losses['l1'] = total.item()

    # Edge BCE derived from edge_logits
    if cfg.loss_edge_weight > 0:
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits, target, reduction='mean')
        losses['edge_bce'] = edge_loss.item()
        total += cfg.loss_edge_weight * edge_loss

    return total, losses


# ============================================================================
# 6. DATASET
# ============================================================================

class DFEDDataset(Dataset):
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, data_dir, cfg, train_mode=True):
        self.cfg = cfg
        self.train_mode = train_mode

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
        print(f"[DFED Dataset] {len(self.samples)} samples")

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

        return {
            'images': torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            'labels': torch.from_numpy(mask[np.newaxis, ...].copy()).float(),
            'file_name': stem + '.png'
        }

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
    print("DFED Training — Distance Field Edge Detector")
    print("=" * 60)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    full_ds = DFEDDataset(cfg.data_dir, cfg, train_mode=True)
    n_val = max(1, int(len(full_ds) * cfg.val_split))
    n_train = len(full_ds) - n_val
    train_idx, val_idx = random_split(
        range(len(full_ds)), [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed))

    class SubDS:
        def __init__(self, ds, indices, train_mode):
            self.ds, self.indices, self.tm = ds, indices, train_mode
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            om = self.ds.train_mode
            self.ds.train_mode = self.tm
            item = self.ds[self.indices[i]]
            self.ds.train_mode = om
            return item

    train_ds = SubDS(full_ds, train_idx.indices, True)
    val_ds = SubDS(full_ds, val_idx.indices, False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_ldr = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ldr = DataLoader(val_ds, 1, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    model = DFED(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.lr_scheduler == "cosine":
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * len(train_ldr), eta_min=cfg.lr_min)
    else:
        sch = optim.lr_scheduler.MultiStepLR(opt, [m * len(train_ldr) for m in cfg.lr_milestones], cfg.lr_gamma)

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
        model.train()
        train_loss = 0
        for bid, batch in enumerate(train_ldr):
            imgs = batch['images'].to(device)
            lbls = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=cfg.fp16):
                outs = model(imgs)
                loss, ld = dfed_loss(outs, lbls, cfg)

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

            if cfg.lr_scheduler in ("cosine",):
                sch.step()

            train_loss += loss.item()
            if bid % cfg.log_interval == 0:
                cstr = ' | '.join(f'{k}:{v:.4f}' for k, v in ld.items())
                print(f"  [E{epoch:3d} B{bid:4d}/{len(train_ldr):4d}] "
                      f"L:{loss.item():.4f} ({cstr}) LR:{opt.param_groups[0]['lr']:.2e}")

        train_loss /= len(train_ldr)
        print(f"  Train Loss: {train_loss:.4f}")

        if cfg.val_interval > 0 and (epoch + 1) % cfg.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_ldr:
                    imgs = batch['images'].to(device)
                    lbls = batch['labels'].to(device)
                    outs = model(imgs)
                    loss, _ = dfed_loss(outs, lbls, cfg)
                    val_loss += loss.item()
            val_loss /= len(val_ldr)
            print(f"  Val Loss: {val_loss:.4f}")

            if cfg.lr_scheduler == "plateau":
                sch.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'best_val_loss': best_val, 'config': cfg},
                           os.path.join(cfg.checkpoint_dir, 'best_model.pth'))

        if (epoch + 1) % cfg.save_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'best_val_loss': best_val, 'config': cfg},
                       os.path.join(cfg.checkpoint_dir, f'epoch_{epoch:03d}.pth'))

    torch.save({'epoch': cfg.epochs, 'model_state_dict': model.state_dict(), 'config': cfg},
               os.path.join(cfg.checkpoint_dir, 'final_model.pth'))
    print(f"\nDone. Best val: {best_val:.4f}")


# ============================================================================
# 8. INFERENCE
# ============================================================================

def load_dfed(checkpoint_path, device, cfg=None):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'config' in ckpt:
        cfg = ckpt['config']
    elif cfg is None:
        cfg = DFEDConfig()
    model = DFED(cfg).to(device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def save_img(tensor, path, original_shape=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
    img = np.squeeze(tensor)
    img = np.uint8((img - img.min()) / (img.max() - img.min() + 1e-8) * 255)
    if original_shape and img.shape != original_shape:
        img = cv2.resize(img, (original_shape[1], original_shape[0]))
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

    # Save distance field
    dist = outs['dist'][0, 0].cpu().numpy()
    save_img(dist, os.path.join(output_dir, f'{stem}_dist.png'), orig)

    # Save edge probability (sigmoid of edge logits)
    edge_prob = torch.sigmoid(outs['edge'])[0, 0].cpu().numpy()
    save_img(1 - edge_prob, os.path.join(output_dir, f'{stem}_edge.png'), orig)

    # Extract valleys → single-pixel skeleton
    if cfg.edge_mode == "valley":
        valleys = find_valleys(dist, threshold=cfg.edge_threshold)
        cv2.imwrite(os.path.join(output_dir, f'{stem}_skeleton.png'),
                    valleys * 255)

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
    print("DFED — Distance Field Edge Detector")
    print("=" * 60)
    model = DFED(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n:,}")
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    model.eval()
    with torch.no_grad():
        outs = model(x)
    for k, v in outs.items():
        print(f"  {k}: {list(v.shape)}")
    print(f"\n  Base channels: {cfg.base_channels}, Depth: {cfg.depth}")
    print(f"  σ for edge prob: {cfg.dist_sigma}")
    print(f"  Edge mode: {cfg.edge_mode}")


def parse_args():
    p = argparse.ArgumentParser(description='DFED')
    p.add_argument('--mode', required=True, choices=['train', 'infer', 'info'])
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--checkpoint_dir', default='./checkpoints')
    p.add_argument('--result_dir', default='./results')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--resume_from', default='')
    p.add_argument('--image_path', default='')
    p.add_argument('--image_dir', default='')
    p.add_argument('--checkpoint', default='./checkpoints/best_model.pth')
    p.add_argument('--edge_mode', default='valley', choices=['valley', 'prob'])
    p.add_argument('--edge_threshold', type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DFEDConfig()
    for k, v in vars(args).items():
        if k in DFEDConfig.__dataclass_fields__:
            setattr(cfg, k, v)

    if args.mode == 'info':
        info(cfg)
    elif args.mode == 'train':
        train(cfg)
    elif args.mode == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_dfed(args.checkpoint, device, cfg)
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
        elif IDE_MODE == 'infer':
            sys.argv.extend(['--checkpoint', IDE_CHECKPOINT, '--result_dir', IDE_RESULT_DIR])
            if IDE_IMAGE_PATH:
                sys.argv.extend(['--image_path', IDE_IMAGE_PATH])
            if IDE_IMAGE_DIR:
                sys.argv.extend(['--image_dir', IDE_IMAGE_DIR])
    main()
