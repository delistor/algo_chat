# TEED-SEM 架构文档

> SEM 半导体量测边缘检测 — 全部架构的完整记录
> 2026-07-26

---

## 目录

1. [文件清单](#1-文件清单)
2. [共同设计原则](#2-共同设计原则)
3. [ETEED — 增强注意力 + 细线头](#3-eteed)
4. [PiDiNet-SEM — 像素差分卷积](#4-pidinet-sem)
5. [DFED — 距离场回归](#5-dfed)
6. [RefineNet — 两阶段精炼](#6-refinenet)
7. [SubPixel-SEM — 亚像素精度架构](#7-subpixel-sem)
8. [TEED Metrology — 后处理骨架化](#8-teed-metrology)
9. [训练数据格式](#9-训练数据格式)
10. [对比速查表](#10-对比速查表)
11. [推荐实验方案](#11-推荐实验方案)

---

## 1. 文件清单

| 文件 | 行数 | 类型 | 用途 |
|------|------|------|------|
| `teed_sem.py` | 1602 | 完整模型 | ETEED：TEED + ECA注意力 + Thin Head |
| `pidinet_sem.py` | 928 | 完整模型 | PiDiNet-SEM：CDC差分卷积 + 方向梯度输入 |
| `dfed_sem.py` | 760 | 完整模型 | DFED：U-Net + 距离场回归 + Valley提取 |
| `refine_sem.py` | 884 | 完整模型 | RefineNet：TEED → 精炼网络两阶段 |
| `subpixel_sem.py` | 898 | 完整模型 | **亚像素精度**：4种架构(HR/ASP/Offset/Combined) |
| `teed_metrology.py` | 894 | 后处理工具 | 骨架化流水线 + OpenCV量测 + Tversky Loss |
| `ted.py` | 298 | 原始模型 | 原始TEED（基线参考） |
| `ARCHITECTURE.md` | — | 文档 | 本文件 |

每个文件都是**完全自包含**的——包含模型定义、数据加载、训练循环、推理、CLI入口、IDE模式。

---

## 2. 共同设计原则

### 2.1 代码结构

每个文件遵循统一的结构：

```
文件
├── IDE Settings（文件顶部，VSCode直接Run）
├── Imports
├── 1. Configuration（dataclass，所有超参数）
├── 2. Activation（Smish）
├── 3. Model Architecture
│   ├── 子模块（ConvBlock, Attention, Heads...）
│   └── 主模型类
├── 4. Loss Functions
├── 5. Dataset（继承 torch.utils.data.Dataset）
├── 6. Training（train函数，完整训练循环）
├── 7. Inference（load + infer函数）
├── 8. CLI（argparse + main）
└── __main__（IDE模式自动转发参数）
```

### 2.2 统一CLI接口

```bash
# 所有文件使用相同的命令模式：
python <file>.py --mode train --data_dir ./data --epochs 20
python <file>.py --mode infer --image_path ./test.jpg
python <file>.py --mode info
```

### 2.3 IDE模式（VSCode / PyCharm）

每个文件顶部有 `IDE_*` 变量。修改 `IDE_MODE = "train"` 后直接按 Run：

```python
# 文件顶部，修改这几行即可在IDE中直接运行
IDE_MODE = "train"          # train | infer | info
IDE_DATA_DIR = "./data"     # 数据路径
IDE_EPOCHS = 10             # 训练轮数
IDE_LR = 8e-4               # 学习率
IDE_IMG_SIZE = 352          # 图片尺寸
IDE_BATCH_SIZE = 8          # 批次大小
```

### 2.4 数据格式

所有模型共用相同的数据格式：`data/raw/` + `data/mask/`

```
data/
├── raw/           # 原始图像（SEM或自然图像）
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── mask/          # 边缘标注（白色边缘，黑色背景）
    ├── img_001.png
    ├── img_002.png
    └── ...
```

---

## 3. ETEED

### 3.1 文件：`teed_sem.py`

### 3.2 设计理念

在原始TEED的基础上做**最小侵入式增强**：
- 保持58K参数的轻量骨架
- 加入ECA注意力（0额外参数）
- 加入可选的Sobel梯度输入通道
- 加入Thin Edge Head（直接用骨架化GT训练）

### 3.3 架构图

```
输入: (B, 3, H, W)  RGB图像
    │
    ├── [可选] Sobel梯度幅度 → 变成4通道输入
    │
    ▼
┌─────────────────────────────────────────┐
│           TEED Backbone (58K)            │
│                                          │
│  block_1: DoubleConv(3/4→16, stride=2)  │
│    ├── ECA(16) ← 通道注意力              │
│    └── side_1 → skip connection          │
│                                          │
│  block_2: DoubleConv(16→32)              │
│    ├── ECA(32)                           │
│    └── maxpool → add skip1               │
│                                          │
│  dblock_3: DenseBlock(1层, 32→48)       │
│    └── ECA(48)                           │
│                                          │
│  UpConv × 3 → out1, out2, out3          │
│  DoubleFusion(cat[o1,o2,o3]) → fused    │
└─────────────────────────────────────────┘
    │
    ├──► Edge Head: fused → sigmoid → 边缘概率 (BCE/Tversky)
    │
    ├──► Thin Head: [fused, coarse_edge] → Conv3→Conv3→Conv1
    │         ↓
    │    细线logits (用骨架化GT + Dice Loss训练)
    │
    └──► Dist Head: fused → Conv3→Conv3→Conv1→sigmoid
              ↓
         距离场 [0,1] (用距离变换GT + L1训练)
```

### 3.4 关键子模块

#### ECA (Efficient Channel Attention)

```
输入: (B, C, H, W)
  → Global Avg Pool → (B, C)
  → Conv1d(k=k_adaptive, groups=1) → (B, C)  ← 跨通道交互，0额外参数
  → Sigmoid → (B, C, 1, 1)
  → 乘回原始特征
```

自适应核大小：`k = |log2(C)/2 + 0.5|_odd`，如 C=16 → k=3, C=32 → k=3, C=48 → k=5

#### ThinEdgeHead

```
输入: fused_features(3ch) + coarse_edge_prob(1ch) = 4ch
  → CDC(4→32, k=3) → Smish
  → CDC(32→16, k=3) → Smish
  → Conv(16→1, k=1) → thin_logits
```

**训练目标**: Zhang-Suen骨架化的GT边缘（单像素宽）
**推理**: thin_logits → sigmoid → threshold → 直接得到单像素骨架

### 3.5 Loss设计

```
Total Loss = 1.0 × FocalTversky(edge, GT, α=0.7, β=0.3, γ=0.75)
           + 1.0 × Dice(thin, skeleton_GT)
           + 0.5 × L1(dist, distance_transform_GT)
           + 0.5 × Σ BDCN(side_output_i, GT)
```

- **Focal Tversky**: α=0.7 强惩罚FP（噪声线），γ=0.75聚焦难分样本
- **Dice Loss**: 解决骨架边缘的极端类别不平衡（~0.1%正样本）
- **Distance L1**: 近边缘像素加权11x，远边缘像素1x
- **BDCN**: 多尺度深度监督

### 3.6 配置关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_attention` | True | ECA注意力开关 |
| `use_gradient_input` | True | Sobel梯度通道开关 |
| `use_thin_head` | True | 细线头开关 |
| `use_dist_head` | False | 距离场头开关 |
| `loss_tversky_alpha` | 0.7 | FP惩罚（越大越干净，但可能漏检） |
| `loss_tversky_beta` | 0.3 | FN惩罚（越大越不漏，但可能多噪） |

---

## 4. PiDiNet-SEM

### 4.1 文件：`pidinet_sem.py`

### 4.2 设计理念

完全不同于TEED的架构路线——用**像素差分卷积(CDC)**替代标准卷积。
CDC迫使卷积核学习像素间的**差异**而非绝对值，天然对边缘敏感。

### 4.3 CDC数学原理

标准卷积：
```
y(p0) = Σ w(pn) · x(p0 + pn)
```
→ 学习的是**强度模式**（"这个纹理长什么样"）

差分卷积（强制核零均值）：
```
y(p0) = Σ w(pn) · (x(p0 + pn) - x(p0))
```
→ 学习的是**梯度模式**（"边缘在哪里，多强"）

CDC（可学习组合）：
```
y = θ · vanilla_conv(x) + (1-θ) · difference_conv(x)
```
θ ∈ [0,1] 是每层独立学习的参数。边缘相关层会学到小的θ（倾向差分模式）。

### 4.4 架构图

```
输入: (B, 3, H, W)  RGB图像
    │
    ├── DirectionalGradientExtractor → 4个方向梯度通道
    │   (0°, 45°, 90°, 135° Prewitt核)
    │
    ▼  合并为7通道: (B, 7, H, W)
    │
┌─────────────────────────────────────────────┐
│            CDC Encoder (131K)                │
│                                              │
│  stem: CDCDoubleBlock(7→16, stride=2)       │
│    → (B, 16, H/2, W/2) ──→ side1 (2x up)   │
│                                              │
│  block1: CDCDoubleBlock(16→32)               │
│    → maxpool → (B, 32, H/4, W/4) ─→ side2   │
│                                              │
│  block2: CDCDoubleBlock(32→48)               │
│    → maxpool → (B, 48, H/8, W/8) ─→ side3   │
│                                              │
│  block3: CDCDoubleBlock(48→64)               │
│    → (B, 64, H/8, W/8) ──→ side4 (8x up)    │
└─────────────────────────────────────────────┘
    │
    ▼  cat[side1, side2, side3, side4] = (B, 4, H, W)
    │
    ├──► Fusion: CDC(4→16) → Conv(16→1) → edge概率
    │
    └──► Thin Head: [fused_cat, edge_prob] → CDC→CDC→Conv
              ↓
         细线logits (骨架GT + Dice Loss)
```

### 4.5 CDC实现细节

```python
class CDC(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        self.weight = Parameter(randn(out_ch, in_ch, k, k))  # 共享权重
        self.theta = Parameter(tensor(0.5))                    # 学习组合比

    def forward(self, x):
        # Vanilla conv
        out_v = conv2d(x, self.weight)

        # Difference conv: 强制核零均值
        zero_mean_w = self.weight - self.weight.mean(dim=(2,3), keepdim=True)
        out_d = conv2d(x, zero_mean_w)

        # 可学习组合
        θ = sigmoid(self.theta)
        return θ·out_v + (1-θ)·out_d
```

### 4.6 方向梯度提取

```
输入: RGB (B,3,H,W)
  → grayscale (0.299R + 0.587G + 0.114B)
  → 4个Prewitt方向核卷积:
     k0:  [[-1,0,1],[-1,0,1],[-1,0,1]]    0° (水平边缘)
     k45: [[0,1,1],[-1,0,1],[-1,-1,0]]    45°
     k90: [[-1,-1,-1],[0,0,0],[1,1,1]]    90° (垂直边缘)
     k135:[[1,1,0],[1,0,-1],[0,-1,-1]]   135°
  → |响应| → 归一化 → (B,4,H,W)
```

### 4.7 Loss

与ETEED相同：FocalTversky(edge) + Dice(thin) + BDCN(side_outputs)

### 4.8 为什么适合SEM

SEM图像的边缘通常是**线/空间结构的几何边界**，具有明确的方向性（水平/垂直的金属线、接触孔边界等）。CDC的方向梯度敏感性天然匹配这种几何特征。

---

## 5. DFED

### 5.1 文件：`dfed_sem.py`

### 5.2 设计理念

**从分类问题转化为回归问题**。边缘检测的本质困难之一是：边缘像素 vs 非边缘像素的分类极端不平衡（边缘通常<1%）。距离场回归优雅地绕过了这个问题：

- 每个像素预测一个**连续值**（到最近边缘的归一化距离）
- 0 = 在边缘上，1 = 远离所有边缘
- 边缘 = 距离场的**局部极小值**（山谷）
- 找到山谷 → 天然的**单像素宽**骨架线

### 5.3 架构图

```
输入: (B, 3, H, W)
    │
    ▼
┌─────────────────────────────────────┐
│         U-Net (492K)                │
│                                      │
│  Encoder:                            │
│    enc1: Conv(3→16) → maxpool        │
│    enc2: Conv(16→32) → maxpool       │
│    enc3: Conv(32→64) → maxpool       │
│    bottleneck: Conv(64→128)          │
│                                      │
│  Decoder (带skip connections):       │
│    dec3: up(128→64) + cat(enc3)     │
│    dec2: up(64→32) + cat(enc2)      │
│    dec1: up(32→16) + cat(enc1)      │
│    final up → (B, 16, H, W)         │
└─────────────────────────────────────┘
    │
    ├──► Dist Head: Conv→Smish→Conv→Sigmoid
    │         ↓
    │    距离场 [0,1]: 0=边缘, 1=远离边缘
    │
    └──► Edge Head: [features, dist] → Conv→Smish→Conv
              ↓
         边缘概率 logits
```

### 5.4 距离场GT生成

```python
def compute_gt_distance(binary_edge_mask):
    # 对边缘mask取反，计算每个像素到最近边缘的欧氏距离
    dist = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, ...)
    # 归一化到 [0, 1]
    dist = dist / dist.max()
    return dist  # 0=边缘, 1=远离边缘
```

### 5.5 边缘提取（推理时）

**Valley Detection（山谷检测）**：

```
输入: 预测的距离场 dist(y, x) ∈ [0, 1]
对于每个像素:
  比较其值与3×3邻域所有像素
  如果 dist[p] < threshold AND dist[p] <= 所有邻居:
    → 标记为边缘（山谷底部）
```

这天然产生**单像素宽**的边缘，因为在任何3×3窗口内最多只有一个山谷。

**或者用概率公式**：
```
edge_prob = exp(-dist² / (2·σ²))
```
σ=0.05时，dist=0→p=1, dist=0.1→p=0.135, dist=0.5→p≈0

### 5.6 Loss设计

```
Total Loss = 1.0 × L1_weighted(dist_pred, dist_GT)
           + 0.1 × GradientConsistency(dist_pred)
           + 0.3 × BCE(edge_logits, GT)
```

**加权L1**：边缘邻近像素（gt_dist < 0.1）权重11x，远处权重1x
**梯度一致性**：鼓励距离场的梯度幅度在边缘附近较大（平滑→锐利过渡）

### 5.7 优劣势

| 优势 | 劣势 |
|------|------|
| 回归比分类稳定（无类别不平衡） | 参数量大（U-Net 492K） |
| 距离场提供丰富的空间信息 | 山谷检测对噪声敏感 |
| 边缘宽度天然为1像素 | 需要调σ/threshold |
| 可解释性强（距离=置信度） | 训练收敛较慢 |

---

## 6. RefineNet

### 6.1 文件：`refine_sem.py`

### 6.2 设计理念

**分而治之**。边缘检测的两个子问题本质上不同：
1. **检出(Detection)**：哪里有边缘？→ 需要大的感受野，语义理解
2. **细化(Refinement)**：边缘长什么样？→ 局部操作，类似NMS+骨架化

Stage 1 解决检出，Stage 2 解决细化。Stage 2 非常轻量（仅16K参数），训练数据需求极低。

### 6.3 架构图

```
输入: (B, 3, H, W)  RGB图像
    │
    ▼
┌──────────────────────────────────┐
│  Stage 1: TEED (58K, 可冻结)      │
│  → coarse edge logits            │
│  → sigmoid → coarse_prob (B,1,H,W)│
└──────────────────────────────────┘
    │
    │  准备RefineNet输入:
    │  ├── image_norm: RGB归一化到[0,1]
    │  ├── coarse_prob: 粗边缘概率
    │  └── grad_mag: Sobel梯度幅度
    │
    ▼  合并为5通道: (B, 5, H, W)
    │
┌──────────────────────────────────┐
│  Stage 2: RefineNet (16K)        │
│                                   │
│  Conv(5→32) → Smish              │
│  Conv(32→32) → Smish             │
│  Conv(32→16) → Smish + skip      │
│  Conv(16→1) → thin_logits        │
└──────────────────────────────────┘
    │
    ▼
输出: (B, 1, H, W) thin_logits
  → sigmoid → threshold → 单像素骨架线
```

### 6.4 RefineNet子模块

```python
class RefineNet(nn.Module):
    """
    输入: [RGB(3), coarse_edge(1), gradient_mag(1)] = 5 channels
    输出: 细化后的边缘logits

    4层CNN，带残差连接。
    参数: 15,889 (仅16K)
    """
    conv1: Conv2d(5 → 32, k=3)    # 288 params
    conv2: Conv2d(32 → 32, k=3)   # 9,248 params
    conv3: Conv2d(32 → 16, k=3)   # 4,624 params
    skip:  Conv2d(32 → 16, k=1)   # 528 params  ← 残差
    conv4: Conv2d(16 → 1, k=1)    # 17 params
    Total: ~16K
```

### 6.5 两阶段训练

```
Phase 1: 训练 Stage 1（TEED）
  - 冻结 Stage 2
  - Loss: FocalTversky(α=0.7) on coarse output
  - Epochs: cfg.epochs (默认10)

Phase 2: 训练 Stage 2（RefineNet）
  - 冻结 Stage 1（可选，推荐）
  - Loss: Dice + BCE on thin output vs skeleton GT
  - Epochs: cfg.stage2_epochs (默认20)
  - 学习率: 2× Phase 1 (更快收敛)
```

### 6.6 数据效率最高的原因

- Stage 2 只有16K参数 → 极低过拟合风险
- Stage 2 的输入已经包含粗边缘概率 → 任务是"修正"而非"从零学习"
- 骨架GT提供精确的像素级监督 → 直接学习细化映射
- 可以冻结Stage 1 → 只训16K参数，几十张图即可

### 6.7 关键配置

| 参数 | 默认 | 说明 |
|------|------|------|
| `train_stage1` | True | 是否从头训TEED |
| `freeze_stage1` | True | Stage 2训练时冻结Stage 1 |
| `stage1_checkpoint` | "" | 预训练TEED权重路径 |
| `stage2_epochs` | 20 | RefineNet训练轮数（建议多于Stage1） |

---

## 7. SubPixel-SEM — 亚像素精度边缘检测

### 7.1 文件：`subpixel_sem.py`

### 7.2 四种亚像素架构

| 架构 | 原理 | 精度 | 参数量 (S=2) |
|------|------|------|---------------|
| **HR-Edge** | PixelShuffle超分 → 2×/4×边缘图 | 1/S pixel | 149K |
| **ASP** | 距离场 + autograd Newton ∇d=0 | 理论无界 | 176K |
| **SP-Offset** | 学习 (Δx,Δy) 偏移回归 | ~0.1 px | 181K |
| **Combined** | HR-Edge + ASP联合精炼 | <0.1 px | 172K |

### 7.3 HR-Edge：PixelShuffle超分

```
输入 H×W → Encoder → Fusion → bilinear up to H → PixelShuffle scale× → S×H × S×W
```
2× = 0.5px精度, 4× = 0.25px精度。简单直接、参数量低。

### 7.4 ASP：解析Newton法

核心：神经网络学到的距离场 d(x,y) 是**连续可微函数**。
对每个离散边缘点做 Newton 迭代求解 ∇d(x*,y*)=0，收敛到亚像素精度。
使用 `grid_sample(bicubic) + autograd.grad()` 实现可微坐标优化。

### 7.5 SP-Offset：偏移回归

Edge Head 输出边缘概率，Offset Head 输出 tanh(·)×0.5 ∈ [-0.5,0.5]²。
真实边缘位置 = (x+Δx, y+Δy)。用高分辨率GT的重心计算训练目标。

### 7.6 Combined：HR + ASP

HR-Edge 先输出 2× 高分辨率离散边缘，
ASP Newton 再将每个边缘点精炼到连续亚像素坐标 → 导出为 CSV。

---

## 8. TEED Metrology

### 7.1 文件：`teed_metrology.py`

### 7.2 定位

这不是模型文件，而是**后处理 + 量测工具**。用于：
1. 对任何边缘概率图做骨架化后处理
2. 用OpenCV做像素级量测（线宽、LER、CD）
3. 提供Tversky Loss供其他模型训练使用

### 7.3 后处理流水线

```
边缘概率图 [0,1] float
    │
    ▼  Step 1: NMS（非极大值抑制）
    │   沿梯度方向保留局部极大值，边缘从5-15px细化到1-2px
    │
    ▼  Step 2: 自适应双阈值
    │   high_thresh = max_val × 0.6
    │   low_thresh = max_val × 0.3
    │   → strong(高阈值) + weak(低阈值)
    │
    ▼  Step 3: 滞后连接（Hysteresis）
    │   BFS从strong出发，连接相邻weak → 保持连续性
    │
    ▼  Step 4: Zhang-Suen骨架化
    │   迭代删除边界像素 → 精确单像素宽
    │
    ▼  Step 5: 断线连接
    │   找端点（8邻域仅有1个邻居）→ Bresenham直线连接 ≤max_gap的端点对
    │
    ▼  Step 6: 短段滤波
    │   去除 <min_length 的连通分量 → 过滤噪声
    │
    ▼
输出: uint8 单像素骨架 (255=边缘, 0=背景)
```

### 7.4 量测工具

| 函数 | 测量内容 | SEM应用 |
|------|----------|---------|
| `measure_line_width()` | 边缘骨架每点的局部线宽 | Line Width |
| `measure_edge_roughness()` | LER 3σ / RMS | 线边粗糙度 |
| `measure_cd()` | 边缘对之间的距离 | Critical Dimension |
| `zhang_suen_skeleton()` | 骨架化 | GT制备 |
| `tversky_loss()` | Tversky Loss | 训练用 |
| `focal_tversky_loss()` | Focal Tversky | 训练用 |

---

## 8. 训练数据格式

### 8.1 目录结构

```
data/
├── raw/                    # 原始图像
│   ├── sem_001.png         # SEM图像或其他图像
│   ├── sem_002.png
│   └── ...
└── mask/                   # 边缘标注（同名文件）
    ├── sem_001.png         # 白色边缘(255) + 黑色背景(0)
    ├── sem_002.png
    └── ...
```

### 8.2 标注要求

- **格式**: 灰度PNG，白色(255) = 边缘，黑色(0) = 非边缘
- **线宽**: 可以较粗（3-5px），训练时自动骨架化为单像素GT
- **文件名**: raw和mask中同名文件自动配对
- **支持格式**: .jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp

### 8.3 数据增强（所有文件统一）

训练时自动应用：
- 随机裁剪（40%概率，最小256px）
- 随机旋转（±15°）
- 水平/垂直翻转（50%概率）
- 亮度/对比度抖动（±20%）
- 边缘增强（GT边缘值+0.2）

---

## 9. 对比速查表

| 特性 | ETEED | PiDiNet-SEM | DFED | RefineNet | SubPixel-HR | SubPixel-ASP | TEED原始 |
|------|-------|-------------|------|-----------|-------------|--------------|----------|
| **文件** | teed_sem | pidinet_sem | dfed_sem | refine_sem | subpixel_sem | subpixel_sem | ted.py |
| **参数量** | 65K | 131K | 492K | 75K | 149K | 176K | 59K |
| **核心卷��** | 标准Conv | **CDC差分** | 标准Conv | 标准Conv | **PixelShuffle** | 标准Conv | 标准Conv |
| **注意力** | **ECA** | 无 | U-Net skip | 无 | 无 | 无 | 无 |
| **输入通道** | 3/4 | **7 (4方向)** | 3 | 3 | 3 | 3 | 3 |
| **细线方式** | Thin Head | Thin Head | Valley | **Stage2学NMS** | **HR直接输出** | **Newton ∇d=0** | 无 |
| **亚像素精度** | ❌ | ❌ | ❌ | ❌ | ✅ **0.25-0.5px** | ✅ **<0.1px** | ❌ |
| **训练策略** | 端到端 | 端到端 | 端到端 | 两阶段 | 端到端 | 端到端 | 端到端 |
| **数据效率** | 中 | 中 | 中-高 | **极高** | 中 | 中 | 中 |
| **推理速度** | 快 | 快 | 中 | 快 | 中 | 慢(Newton迭代) | 最快 |
| **适合SEM** | ✅ | ✅✅ | ✅✅ | ✅✅✅ | ✅✅ **计量级** | ✅✅✅ **最高精度** | 基线 |

---

## 10. 推荐实验方案

### 10.1 第一轮：快速评估（1天）

```
目标: 找出最有潜力的方法
数据: 20-50张SEM标注图

1. RefineNet: 用TEED预训练权重做Stage1，只训Stage2（16K参数）
   预期: 最快看到细线效果
   python refine_sem.py --mode train --stage1_checkpoint checkpoints/BIPED/7/7_model.pth

2. ETEED: 用TEED预训练权重初始化backbone，微调全部
   预期: 比原始TEED有明显提升
   python teed_sem.py --mode train --pretrained_backbone checkpoints/BIPED/7/7_model.pth

3. 原始TEED微调（基线对比）
   python teed_clean_3.py --mode train
```

### 10.2 第二轮：深度对比（1周）

```
目标: 完整训练所有方法，找最优
数据: 100-500张SEM标注图

1. PiDiNet-SEM: 从头训练（CDC需要从头学，预训练权重不适用）
2. DFED: 从头训练（U-Net，可能需要更多数据）
3. 超参数调优：调整tversky_alpha/beta, lr, epochs
4. 交叉验证：不同SEM图像类型/放大倍率/噪声水平
```

### 10.3 量测验证

```bash
# 对所有方法的输出做量测对比
python teed_metrology.py \
    --checkpoint ./checkpoints/<method>/best_model.pth \
    --input ./sem_test/ \
    --output ./results/<method>/ \
    --measure --scale_nm_per_px 0.25
```

---

## 附录A: 参数计算

| 模型 | 骨干参数 | Head参数 | 总参数 |
|------|----------|----------|--------|
| TEED (原始) | 58,910 | 0 | 58,910 |
| ETEED | 58,910 | Thin: 5,978 | 64,888 |
| ETEED (+grad) | 59,778 | Thin: 5,978 | 65,756 |
| ETEED (+grad+dist) | 59,778 | Thin+Dist: 11,490 | 71,268 |
| PiDiNet-SEM | 126,364 | Thin: 5,081 | 131,445 |
| DFED | 484,450 | Dist+Edge: 7,200 | 491,650 |
| RefineNet S1 | 58,910 | 0 | 58,910 |
| RefineNet S2 | 0 | 15,889 | 15,889 |
| RefineNet Total | 58,910 | 15,889 | 74,799 |

## 附录B: Loss函数公式

### Focal Tversky Loss
```
Tversky = (TP + ε) / (TP + α·FP + β·FN + ε)
FocalTversky = (1 - Tversky)^γ
```
α控制FP惩罚（噪声），β控制FN惩罚（漏检），γ控制难分样本聚焦度。

### Dice Loss
```
Dice = (2·|pred ∩ target| + ε) / (|pred| + |target| + ε)
DiceLoss = 1 - Dice
```
适合极度不平衡的二分类（骨架边缘≈0.1%像素）。

### Distance Field Loss
```
L1_weighted = Σ weight(p) · |dist_pred(p) - dist_GT(p)|
weight(p) = 1 + 10 · 1[dist_GT(p) < 0.1]
```
近边缘像素（<10% max_dist）获得11x权重。

---

*生成时间: 2026-07-26 | 工具: TEED-SEM Research Pipeline*
