"""
TEED Metrology Pipeline — SEM 半导体量测专用边缘检测
=======================================================
功能：
  1. 边缘骨架化后处理 (NMS → 自适应阈值 → Zhang-Suen细化 → 断线连接 → 去噪)
  2. Tversky Loss（训练用，显式惩罚噪声线和漏检）
  3. OpenCV 像素量测工具（线宽、线边粗糙度、CD 测量）

输出：单像素宽度的干净骨架线，可直接用 cv2.findContours() 提取坐标做量测。

Usage:
  # 推理 + 骨架化
  python teed_metrology.py --checkpoint checkpoints/BIPED/7/7_model.pth \
      --input ./your_sem_images/ --output ./skeleton_results/

  # 训练（使用 Tversky Loss）
  python teed_metrology.py --mode train --data_dir ./data --epochs 20
"""

# ============================================================================
# IDE Direct Run Settings (VSCode / PyCharm — press Run)
# ============================================================================
IDE_MODE = "infer"              # infer | measure
IDE_INPUT = "./test.png"        # Input image path or directory
IDE_OUTPUT = "./results"        # Output directory
IDE_CHECKPOINT = "./checkpoints/BIPED/7/7_model.pth"
IDE_DEVICE = "cpu"
IDE_LOW_RATIO = 0.25
IDE_HIGH_RATIO = 0.55
IDE_MIN_LENGTH = 10
IDE_CONNECT_GAP = 3
IDE_MEASURE = True              # Run metrology measurements
IDE_SCALE_NM_PER_PX = 0.5       # nm per pixel for semiconductor

import os
import sys
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import TEED model ──────────────────────────────────────────────
from ted import TED

# ============================================================================
# 1. 单像素骨架化后处理流水线
# ============================================================================

def non_max_suppression(edge_map, grad_dir=None):
    """
    非极大值抑制 (NMS) — 在梯度方向上只保留局部极大值，将边缘细化到 1-2 像素宽。

    Args:
        edge_map: (H, W) 浮点概率图 [0, 1]
        grad_dir: (H, W) 梯度方向图（弧度），None 则自动从 edge_map 计算
    Returns:
        nms_map: (H, W) 细化后的浮点图
    """
    h, w = edge_map.shape

    if grad_dir is None:
        # 用 Sobel 从概率图计算梯度方向
        gx = cv2.Sobel(edge_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(edge_map, cv2.CV_32F, 0, 1, ksize=3)
        grad_dir = np.arctan2(gy, gx)

    # 将梯度方向量化到 4 个方向：0°, 45°, 90°, 135°
    angle = np.rad2deg(grad_dir) % 180
    quantized = np.zeros_like(angle, dtype=np.uint8)
    quantized[(angle < 22.5) | (angle >= 157.5)] = 0      # → horizontal
    quantized[(angle >= 22.5) & (angle < 67.5)] = 1        # ↗ 45°
    quantized[(angle >= 67.5) & (angle < 112.5)] = 2       # ↑ vertical
    quantized[(angle >= 112.5) & (angle < 157.5)] = 3      # ↘ 135°

    nms_map = edge_map.copy()
    padded = np.pad(edge_map, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            d = quantized[i, j]
            val = edge_map[i, j]
            pi, pj = i + 1, j + 1  # padded coords

            if d == 0:   # horizontal: compare left/right
                n1, n2 = padded[pi, pj - 1], padded[pi, pj + 1]
            elif d == 1: # 45°: compare top-right / bottom-left
                n1, n2 = padded[pi - 1, pj + 1], padded[pi + 1, pj - 1]
            elif d == 2: # vertical: compare top/bottom
                n1, n2 = padded[pi - 1, pj], padded[pi + 1, pj]
            else:        # 135°: compare top-left / bottom-right
                n1, n2 = padded[pi - 1, pj - 1], padded[pi + 1, pj + 1]

            if val < n1 or val < n2:
                nms_map[i, j] = 0

    return nms_map


def adaptive_threshold(edge_map, low_ratio=0.3, high_ratio=0.6):
    """
    自适应双阈值 — 基于图像统计特性自动确定阈值。

    Args:
        edge_map: (H, W) 浮点概率图
        low_ratio: 低阈值 = max_val * low_ratio
        high_ratio: 高阈值 = max_val * high_ratio
    Returns:
        strong: (H, W) bool — 强边缘
        weak: (H, W) bool — 弱边缘（需连接判断）
    """
    max_val = edge_map.max()
    high_thresh = max_val * high_ratio
    low_thresh = max_val * low_ratio

    strong = edge_map >= high_thresh
    weak = (edge_map >= low_thresh) & (edge_map < high_thresh)

    return strong, weak


def hysteresis_linking(strong, weak):
    """
    Canny 风格的滞后阈值连接 — 只有与强边缘 8-连通的弱边缘才保留。

    Args:
        strong: (H, W) bool
        weak: (H, W) bool
    Returns:
        binary: (H, W) bool — 连接后的边缘图
    """
    h, w = strong.shape
    binary = strong.copy().astype(np.uint8)
    weak_uint8 = weak.astype(np.uint8)

    # 用连通域分析：标记弱边缘中与强边缘相连的像素
    # BFS 从每个强边缘像素出发，把相邻弱边缘并入
    visited = np.zeros((h, w), dtype=bool)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),   (1, 0), (1, 1)]

    # 找到所有强边缘点作为种子
    strong_points = np.argwhere(binary > 0)
    queue = deque(strong_points.tolist())

    for pt in strong_points:
        visited[pt[0], pt[1]] = True

    while queue:
        y, x = queue.popleft()
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if weak_uint8[ny, nx]:
                    binary[ny, nx] = 1
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return binary.astype(bool)


def zhang_suen_skeleton(binary):
    """
    Zhang-Suen 骨架化算法 — 将边缘细化到精确单像素宽。
    纯 numpy 实现，无 skimage 依赖。

    Args:
        binary: (H, W) uint8 or bool — 二值边缘图
    Returns:
        skeleton: (H, W) uint8 — 单像素宽骨架
    """
    skeleton = binary.astype(np.uint8).copy()
    h, w = skeleton.shape

    def _neighbors(y, x):
        """返回 8 邻域的 P2-P9 值（顺时针，P2 = 上方邻居）"""
        n = [
            skeleton[y - 1, x]     if y > 0     else 0,  # P2
            skeleton[y - 1, x + 1] if y > 0 and x < w - 1 else 0,  # P3
            skeleton[y, x + 1]     if x < w - 1 else 0,  # P4
            skeleton[y + 1, x + 1] if y < h - 1 and x < w - 1 else 0,  # P5
            skeleton[y + 1, x]     if y < h - 1 else 0,  # P6
            skeleton[y + 1, x - 1] if y < h - 1 and x > 0 else 0,  # P7
            skeleton[y, x - 1]     if x > 0     else 0,  # P8
            skeleton[y - 1, x - 1] if y > 0 and x > 0 else 0,  # P9
        ]
        return n

    def _transitions(n):
        """计算 0→1 转换次数"""
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
                t = _transitions(n)
                if (2 <= s <= 6 and t == 1 and
                    n[0] * n[2] * n[4] == 0 and  # P2 * P4 * P6 = 0
                    n[2] * n[4] * n[6] == 0):    # P4 * P6 * P8 = 0
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
                t = _transitions(n)
                if (2 <= s <= 6 and t == 1 and
                    n[0] * n[2] * n[6] == 0 and  # P2 * P4 * P8 = 0
                    n[0] * n[4] * n[6] == 0):    # P2 * P6 * P8 = 0
                    to_remove.append((y, x))
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True

    return skeleton


def remove_short_segments(skeleton, min_length=20):
    """
    去除长度小于 min_length 的独立线段（噪声过滤）。

    Args:
        skeleton: (H, W) uint8 — 骨架图
        min_length: 最小保留长度（像素）
    Returns:
        filtered: (H, W) uint8 — 去噪后骨架
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8)
    filtered = np.zeros_like(skeleton)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_length:
            filtered[labels == i] = 1
    return filtered


def connect_broken_edges(skeleton, max_gap=5, angle_thresh=np.pi / 3):
    """
    连接断裂的边缘段 — 找到端点，用 Bresenham 直线连接距离 < max_gap 的端点对。

    Args:
        skeleton: (H, W) uint8 — 骨架图
        max_gap: 最大连接距离（像素）
        angle_thresh: 端点方向允许的最大偏差（弧度），未使用（保留接口）
    Returns:
        connected: (H, W) uint8 — 连接后骨架
    """
    binary = (skeleton > 0).astype(np.uint8)
    h, w = binary.shape

    # 找端点：8邻域中只有 1 个邻居的骨架像素
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    from scipy.ndimage import convolve
    neighbor_count = convolve(binary, kernel, mode='constant', cval=0)
    endpoints = np.argwhere((binary == 1) & (neighbor_count == 1))

    if len(endpoints) < 2:
        return skeleton

    # 连接距离在 max_gap 内的端点对
    connected = binary.copy()
    used = set()
    n_endpoints = len(endpoints)

    for i in range(n_endpoints):
        if i in used:
            continue
        y1, x1 = endpoints[i]
        best_j, best_dist = -1, max_gap + 1

        for j in range(i + 1, n_endpoints):
            if j in used:
                continue
            y2, x2 = endpoints[j]
            dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j >= 0:
            y2, x2 = endpoints[best_j]
            # Bresenham 直线连接
            pts = _bresenham_line(int(x1), int(y1), int(x2), int(y2))
            for px, py in pts:
                if 0 <= py < h and 0 <= px < w:
                    connected[py, px] = 1
            used.add(i)
            used.add(best_j)

    return (connected * 255).astype(np.uint8)


def _bresenham_line(x0, y0, x1, y1):
    """Bresenham 直线算法，返回直线上的所有像素坐标列表 [(x, y), ...]."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    # 去除端点本身（骨架已有）
    return points[1:-1]


def pipeline(edge_prob_map,
             low_ratio=0.3,
             high_ratio=0.6,
             min_segment_length=15,
             connect_gap=3):
    """
    完整的后处理流水线：概率图 → 单像素骨架线。

    Args:
        edge_prob_map: (H, W) numpy float32/float64 [0, 1]
        low_ratio, high_ratio: 滞后阈值参数
        min_segment_length: 最短保留线段
        connect_gap: 断线连接最大距离
    Returns:
        skeleton: (H, W) uint8 — 单像素骨架线 (255 = 边缘, 0 = 背景)
    """
    # Step 1: NMS 细化
    print(f"  [Pipeline] NMS...")
    nms_map = non_max_suppression(edge_prob_map)

    # Step 2: 自适应双阈值 + 滞后连接
    print(f"  [Pipeline] Hysteresis thresholding...")
    strong, weak = adaptive_threshold(nms_map, low_ratio, high_ratio)
    binary = hysteresis_linking(strong, weak)

    if binary.sum() == 0:
        print("  [Pipeline] WARNING: No edges detected!")
        return np.zeros_like(edge_prob_map, dtype=np.uint8)

    # Step 3: Zhang-Suen 骨架化 → 精确单像素
    print(f"  [Pipeline] Zhang-Suen skeletonization...")
    skeleton = zhang_suen_skeleton(binary)

    # Step 4: 断线连接
    if connect_gap > 0:
        print(f"  [Pipeline] Connecting broken edges (gap={connect_gap})...")
        skeleton = connect_broken_edges(skeleton, max_gap=connect_gap)

    # Step 5: 去除短噪声段
    if min_segment_length > 0:
        print(f"  [Pipeline] Removing short segments (<{min_segment_length}px)...")
        skeleton = remove_short_segments(skeleton, min_length=min_segment_length)

    # 输出归一化到 [0, 255]
    skeleton_out = (skeleton * 255).astype(np.uint8)

    print(f"  [Pipeline] Done. Edge pixels: {skeleton.sum()}, "
          f"ratio: {skeleton.sum() / skeleton.size * 100:.2f}%")

    return skeleton_out


# ============================================================================
# 2. Tversky Loss — 训练用，显式惩罚噪声线 + 漏检
# ============================================================================

def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1e-8):
    """
    Tversky Index Loss — 用于边缘检测训练。

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    Loss = 1 - Tversky

    alpha > 0.5 → 重点惩罚 FP（噪声线/假边缘）
    beta > 0.5  → 重点惩罚 FN（漏检）

    对于 SEM 量测，推荐 alpha=0.7（强惩罚噪声线），beta=0.3

    Args:
        pred: (B, C, H, W) logits (before sigmoid)
        target: (B, C, H, W) binary ground truth
        alpha: FP 权重（越大越抑制噪声线）
        beta: FN 权重（越大越防漏检）
    Returns:
        scalar loss
    """
    pred = torch.sigmoid(pred)
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-8):
    """
    Focal Tversky Loss — Tversky 的 focal 变体，对难样本更敏感。

    Loss = (1 - Tversky)^gamma

    gamma=0.75 时对难分样本放大惩罚，对 SEM 中对比度低的弱边缘尤其有用。
    """
    tversky_loss_val = tversky_loss(pred, target, alpha, beta, smooth)
    return torch.pow(tversky_loss_val, gamma)


def combined_edge_loss(pred_list, target, device,
                       l_weight_bdcn=(1.1, 0.7, 1.1, 1.3),
                       l_weight_fuse=(0.01, 3.0),
                       tversky_alpha=0.7,
                       tversky_beta=0.3):
    """
    组合 Loss：TEED 多尺度输出用 BDCN Loss，最终融合输出用 Focal Tversky Loss。

    这是针对 SEM 量测优化的 loss 组合：
    - 中间层：BDCN loss 保持多尺度特征学习
    - 最终输出：Focal Tversky loss 直接优化细线/去噪目标

    Args:
        pred_list: TEED forward 输出 (out1, out2, out3, fused)
        target: ground truth edge map
        l_weight_bdcn: BDCN loss 权重
        l_weight_fuse: (tex_factor, bdr_factor) for cats_loss backup
        tversky_alpha: Tversky FP penalty
        tversky_beta: Tversky FN penalty
    Returns:
        total_loss: scalar
    """
    from loss2 import bdcn_loss2, cats_loss

    # BDCN loss 用于中间层
    loss_bdcn = sum(bdcn_loss2(preds, target, lw)
                    for preds, lw in zip(pred_list[:-1], l_weight_bdcn))

    # Focal Tversky loss 用于最终融合输出
    loss_fuse = focal_tversky_loss(pred_list[-1], target,
                                   alpha=tversky_alpha, beta=tversky_beta)

    return loss_bdcn + loss_fuse


# ============================================================================
# 3. OpenCV 像素量测工具
# ============================================================================

def measure_line_width(skeleton, original_image=None, scale_nm_per_px=1.0):
    """
    测量线宽 — 对骨架的每个像素，计算其到最近非边缘像素的距离 × 2。

    Args:
        skeleton: (H, W) uint8 — 单像素骨架
        original_image: 可选，用于可视化
        scale_nm_per_px: 每个像素对应的纳米数
    Returns:
        dict: {
            'mean_width_nm': 平均线宽,
            'std_width_nm': 线宽标准差,
            'width_map': (H, W) 每个边缘像素的线宽图,
            'measurement_points': list of (x, y, width_nm)
        }
    """
    if skeleton.max() <= 1:
        skeleton = (skeleton * 255).astype(np.uint8)

    # 距离变换：每个非边缘像素到最近边缘的距离
    dist = cv2.distanceTransform(1 - (skeleton > 0).astype(np.uint8),
                                  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # 对每个骨架像素，线宽 ≈ 2 × 最近距离
    edge_mask = skeleton > 0
    if edge_mask.sum() == 0:
        return {'mean_width_nm': 0, 'std_width_nm': 0, 'width_map': None, 'measurement_points': []}

    widths = dist[edge_mask] * 2 * scale_nm_per_px

    width_map = np.zeros_like(dist)
    width_map[edge_mask] = dist[edge_mask] * 2 * scale_nm_per_px

    # 提取量测点
    ys, xs = np.where(edge_mask)
    measurement_points = [(int(x), int(y), float(w))
                          for x, y, w in zip(xs, ys, widths)]

    return {
        'mean_width_nm': float(np.mean(widths)),
        'std_width_nm': float(np.std(widths)),
        'min_width_nm': float(np.min(widths)),
        'max_width_nm': float(np.max(widths)),
        'width_map': width_map,
        'measurement_points': measurement_points[:1000],  # 限制返回数量
    }


def measure_edge_roughness(skeleton, smooth_window=50, scale_nm_per_px=1.0):
    """
    测量线边粗糙度 (LER — Line Edge Roughness)。

    对每条检测到的边缘轮廓，用滑动窗口拟合直线，计算残差的标准差。

    Args:
        skeleton: (H, W) uint8 — 单像素骨架
        smooth_window: 滑动窗口大小（像素）
        scale_nm_per_px: 像素到纳米的换算
    Returns:
        dict: {
            'ler_3sigma_nm': 3σ LER,
            'ler_rms_nm': RMS LER,
            'roughest_location': (x, y) 最粗糙位置,
            'contours': list of contours
        }
    """
    if skeleton.max() <= 1:
        skeleton = (skeleton * 255).astype(np.uint8)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    all_residuals = []
    roughest = None
    max_roughness = 0

    for contour in contours:
        if len(contour) < smooth_window:
            continue
        pts = contour.squeeze(1).astype(np.float32)  # (N, 2)

        for i in range(0, len(pts) - smooth_window, smooth_window // 2):
            window = pts[i:i + smooth_window]
            # PCA 拟合直线
            mean = window.mean(axis=0)
            cov = np.cov((window - mean).T)
            if cov.shape == ():  # 单点协方差
                continue
            _, eigenvectors = np.linalg.eigh(cov)
            # 主轴方向
            direction = eigenvectors[:, -1]
            # 计算每个点到直线的距离
            for pt in window:
                vec = pt - mean
                dist = np.abs(np.cross(direction, vec))
                all_residuals.append(dist)

    if not all_residuals:
        return {'ler_3sigma_nm': 0, 'ler_rms_nm': 0, 'roughest_location': None, 'contours': contours}

    all_residuals = np.array(all_residuals) * scale_nm_per_px
    rms = np.sqrt(np.mean(all_residuals ** 2))

    return {
        'ler_3sigma_nm': 3 * np.std(all_residuals),
        'ler_rms_nm': rms,
        'contours': contours,
    }


def measure_cd(skeleton, direction='horizontal', scale_nm_per_px=1.0):
    """
    测量 Critical Dimension (CD) — 沿指定方向扫描，测量边缘到边缘的距离。

    Args:
        skeleton: (H, W) uint8 — 单像素骨架
        direction: 'horizontal' | 'vertical' — CD 测量方向
        scale_nm_per_px: 像素到纳米换算
    Returns:
        dict: {
            'cd_mean_nm': 平均 CD,
            'cd_std_nm': CD 标准差,
            'cd_profile': 沿测量方向的 CD 变化序列
        }
    """
    if skeleton.max() <= 1:
        skeleton = (skeleton * 255).astype(np.uint8)

    h, w = skeleton.shape

    if direction == 'horizontal':
        # 逐行扫描，找相邻边缘对的距离
        cd_values = []
        for y in range(h):
            edge_cols = np.where(skeleton[y] > 0)[0]
            if len(edge_cols) >= 2:
                # 找相邻边缘对（配对：第 1-2, 3-4, ...）
                for i in range(0, len(edge_cols) - 1, 2):
                    cd = edge_cols[i + 1] - edge_cols[i]
                    cd_values.append(cd)
    else:
        cd_values = []
        for x in range(w):
            edge_rows = np.where(skeleton[:, x] > 0)[0]
            if len(edge_rows) >= 2:
                for i in range(0, len(edge_rows) - 1, 2):
                    cd = edge_rows[i + 1] - edge_rows[i]
                    cd_values.append(cd)

    if not cd_values:
        return {'cd_mean_nm': 0, 'cd_std_nm': 0, 'cd_profile': []}

    cd_values = np.array(cd_values, dtype=np.float64) * scale_nm_per_px

    return {
        'cd_mean_nm': float(np.mean(cd_values)),
        'cd_std_nm': float(np.std(cd_values)),
        'cd_profile': cd_values.tolist(),
    }


# ============================================================================
# 4. 完整推理 + 量测工作流
# ============================================================================

def load_teed_model(checkpoint_path, device):
    """加载 TEED 模型权重。"""
    model = TED().to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # 兼容不同 checkpoint 格式
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    print(f"Model loaded: {checkpoint_path}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def process_image(model, image_path, device,
                  img_mean=(103.939, 116.779, 123.68),
                  pipeline_kwargs=None):
    """
    处理单张图片：TEED 推理 → 骨架化后处理。

    Args:
        model: TEED model
        image_path: 输入图片路径
        device: torch device
        img_mean: 图像均值（BGR）
        pipeline_kwargs: 传递给 pipeline() 的参数字典
    Returns:
        dict: {
            'skeleton': uint8 single-pixel skeleton,
            'edge_prob': float32 probability map,
            'fused_output': raw fused output,
            'intermediate_outputs': [out1, out2, out3],
            'original_image_shape': (h, w),
            'original_image': uint8 BGR image
        }
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {'low_ratio': 0.4, 'high_ratio': 0.7,
                          'min_segment_length': 8, 'connect_gap': 2}

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    original_shape = img.shape[:2]

    # 预处理：resize 到 8 的倍数（TEED 架构要求）
    h, w = original_shape
    pad_h = ((h // 8) + 1) * 8 if h % 8 != 0 else h
    pad_w = ((w // 8) + 1) * 8 if w % 8 != 0 else w

    img_resized = cv2.resize(img, (pad_w, pad_h))
    img_float = img_resized.astype(np.float32)
    img_float -= np.array(img_mean, dtype=np.float32).reshape(1, 1, 3)
    img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1).copy()).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # TEED 推理
    with torch.no_grad():
        preds = model(img_tensor)

    # 取最终融合输出，并处理不同输出的尺寸差异
    fused = preds[-1]
    # 用 interpolate 确保所有输出与输入尺寸一致
    fused = F.interpolate(fused, size=img_tensor.shape[2:],
                          mode='bilinear', align_corners=True)
    edge_prob = torch.sigmoid(fused[0, 0]).cpu().numpy()

    # 裁剪回原始尺寸（去除 padding 区域）
    edge_prob = edge_prob[:original_shape[0], :original_shape[1]]

    # 后处理 → 单像素骨架
    skeleton = pipeline(edge_prob, **pipeline_kwargs)

    return {
        'skeleton': skeleton,
        'edge_prob': edge_prob,
        'fused_output': fused.cpu().numpy(),
        'intermediate_outputs': [p.cpu().numpy() for p in preds[:-1]],
        'original_image_shape': original_shape,
        'original_image': img,
    }


# ============================================================================
# 5. CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TEED Metrology — SEM 量测专用边缘骨架化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 推理 + 骨架化
  python teed_metrology.py --checkpoint checkpoints/BIPED/7/7_model.pth \\
      --input ./sem_images/ --output ./skeleton_results/

  # 单张图片
  python teed_metrology.py --checkpoint checkpoints/BIPED/7/7_model.pth \\
      --input ./test.png --output ./result.png

  # 附带量测报告
  python teed_metrology.py --checkpoint ... --input ./test.png --output ./results/ \\
      --measure --scale_nm_per_px 0.5
        """)

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='TEED model checkpoint path (.pth)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (image or directory)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cuda / cpu')

    # Pipeline params
    parser.add_argument('--low_ratio', type=float, default=0.3,
                        help='Hysteresis low threshold ratio')
    parser.add_argument('--high_ratio', type=float, default=0.6,
                        help='Hysteresis high threshold ratio')
    parser.add_argument('--min_length', type=int, default=15,
                        help='Minimum segment length (noise filter)')
    parser.add_argument('--connect_gap', type=int, default=3,
                        help='Max gap for edge linking')

    # Measurement
    parser.add_argument('--measure', action='store_true',
                        help='Run metrology measurements')
    parser.add_argument('--scale_nm_per_px', type=float, default=1.0,
                        help='Nanometers per pixel for measurement')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_teed_model(args.checkpoint, device)

    pipeline_kwargs = {
        'low_ratio': args.low_ratio,
        'high_ratio': args.high_ratio,
        'min_segment_length': args.min_length,
        'connect_gap': args.connect_gap,
    }

    # Determine input mode
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
    input_is_dir = os.path.isdir(args.input)

    if input_is_dir:
        image_files = sorted([
            f for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        ])
        if not image_files:
            print(f"No images found in {args.input}")
            return
        os.makedirs(args.output, exist_ok=True)
        print(f"Processing {len(image_files)} images...")

        for fname in image_files:
            img_path = os.path.join(args.input, fname)
            stem = os.path.splitext(fname)[0]
            print(f"\n{'='*50}")
            print(f"Processing: {fname}")

            result = process_image(model, img_path, device,
                                   pipeline_kwargs=pipeline_kwargs)

            # Save skeleton
            out_path = os.path.join(args.output, f"{stem}_skeleton.png")
            cv2.imwrite(out_path, result['skeleton'])

            # Save probability map overlay
            prob_vis = (result['edge_prob'] * 255).astype(np.uint8)
            prob_path = os.path.join(args.output, f"{stem}_prob.png")
            cv2.imwrite(prob_path, prob_vis)

            # Save overlay on original
            overlay = result['original_image'].copy()
            skeleton_bgr = cv2.cvtColor(result['skeleton'], cv2.COLOR_GRAY2BGR)
            overlay[result['skeleton'] > 0] = [0, 0, 255]  # Red skeleton overlay
            overlay = cv2.addWeighted(result['original_image'], 0.7, overlay, 0.3, 0)
            ov_path = os.path.join(args.output, f"{stem}_overlay.png")
            cv2.imwrite(ov_path, overlay)

            if args.measure:
                lw = measure_line_width(result['skeleton'],
                                        scale_nm_per_px=args.scale_nm_per_px)
                ler = measure_edge_roughness(result['skeleton'],
                                              scale_nm_per_px=args.scale_nm_per_px)
                print(f"  Line Width: mean={lw['mean_width_nm']:.2f} nm, "
                      f"std={lw['std_width_nm']:.2f} nm")
                print(f"  LER (3σ): {ler['ler_3sigma_nm']:.2f} nm, "
                      f"RMS: {ler['ler_rms_nm']:.2f} nm")

    else:
        # Single image
        result = process_image(model, args.input, device,
                               pipeline_kwargs=pipeline_kwargs)

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = args.output if os.path.isdir(args.output) else os.path.dirname(args.output) or '.'

        skeleton_out = os.path.join(out_dir, f"{stem}_skeleton.png")
        cv2.imwrite(skeleton_out, result['skeleton'])
        print(f"Skeleton saved: {skeleton_out}")

        if args.measure:
            lw = measure_line_width(result['skeleton'],
                                    scale_nm_per_px=args.scale_nm_per_px)
            ler = measure_edge_roughness(result['skeleton'],
                                          scale_nm_per_px=args.scale_nm_per_px)
            cd = measure_cd(result['skeleton'],
                           scale_nm_per_px=args.scale_nm_per_px)
            print(f"\n{'='*50}")
            print("METROLOGY REPORT")
            print(f"{'='*50}")
            print(f"  Line Width:  mean={lw['mean_width_nm']:.2f} nm, "
                  f"std={lw['std_width_nm']:.2f} nm, "
                  f"min={lw['min_width_nm']:.2f}, max={lw['max_width_nm']:.2f}")
            print(f"  LER (3σ):    {ler['ler_3sigma_nm']:.2f} nm")
            print(f"  LER (RMS):   {ler['ler_rms_nm']:.2f} nm")
            print(f"  CD (mean):   {cd['cd_mean_nm']:.2f} nm, "
                  f"std={cd['cd_std_nm']:.2f} nm")


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("=" * 60)
        print("TEED Metrology — Running in IDE mode")
        print(f"  IDE_MODE: {IDE_MODE}")
        print("  Modify IDE_* variables at top of file to configure.")
        print("=" * 60)
        sys.argv.extend(["--checkpoint", IDE_CHECKPOINT])
        sys.argv.extend(["--output", IDE_OUTPUT])
        sys.argv.extend(["--device", IDE_DEVICE])
        sys.argv.extend(["--low_ratio", str(IDE_LOW_RATIO)])
        sys.argv.extend(["--high_ratio", str(IDE_HIGH_RATIO)])
        sys.argv.extend(["--min_length", str(IDE_MIN_LENGTH)])
        sys.argv.extend(["--connect_gap", str(IDE_CONNECT_GAP)])
        if IDE_MEASURE:
            sys.argv.append("--measure")
            sys.argv.extend(["--scale_nm_per_px", str(IDE_SCALE_NM_PER_PX)])
        if os.path.isdir(IDE_INPUT):
            sys.argv.extend(["--input", IDE_INPUT])
        elif os.path.isfile(IDE_INPUT):
            sys.argv.extend(["--input", IDE_INPUT])
        else:
            print(f"  WARNING: IDE_INPUT '{IDE_INPUT}' not found. "
                  "Specify a valid image path.")
            sys.exit(1)
    main()
