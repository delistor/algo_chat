"""
XGBoost 图像分类器 —— 训练推理一体，单文件脚本
适用场景: 32x32 灰度/二值图，data/ 下按类别分子文件夹
用法:
  训练:  python xgb_classifier.py train
  推理:  python xgb_classifier.py predict <image_path>
  评估:  python xgb_classifier.py eval
"""

import os
import sys
import pickle
import argparse
from glob import glob
from collections import Counter

import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# ─── 配置 ───────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_PATH = "xgb_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = 32

os.environ.setdefault("LOKY_MAX_CPU_COUNT", os.cpu_count() or "4")


# ═══════════════════════════════════════════════════════════════
#  特征工程 —— 全手动构造
# ═══════════════════════════════════════════════════════════════

def _glcm_matrix(img_q: np.ndarray, levels: int, dy: int, dx: int):
    """单方向灰度共生矩阵 (向量化)"""
    h, w = img_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    if dy >= 0 and dx >= 0:
        src_h, src_w = h - dy, w - dx
        if src_h <= 0 or src_w <= 0:
            return glcm
        src = img_q[:src_h, :src_w].ravel()
        dst = img_q[dy:dy + src_h, dx:dx + src_w].ravel()
    elif dy >= 0 and dx < 0:
        src_h, src_w = h - dy, w + dx
        if src_h <= 0 or src_w <= 0:
            return glcm
        src = img_q[:src_h, -dx:-dx + src_w].ravel()
        dst = img_q[dy:dy + src_h, :src_w].ravel()
    elif dy < 0 and dx >= 0:
        src_h, src_w = h + dy, w - dx
        if src_h <= 0 or src_w <= 0:
            return glcm
        src = img_q[-dy:-dy + src_h, :src_w].ravel()
        dst = img_q[:src_h, dx:dx + src_w].ravel()
    else:
        src_h, src_w = h + dy, w + dx
        if src_h <= 0 or src_w <= 0:
            return glcm
        src = img_q[-dy:-dy + src_h, -dx:-dx + src_w].ravel()
        dst = img_q[:src_h, :src_w].ravel()

    np.add.at(glcm.ravel(), src * levels + dst, 1)
    glcm += glcm.T
    glcm /= glcm.sum() + 1e-8
    return glcm


def _glcm_features(img: np.ndarray, levels: int = 8) -> list:
    """
    GLCM Haralick 纹理特征: 对比度、相异性、同质性、能量、相关性 × 多种方向距离
    """
    img_f = img.astype(np.float64)
    img_q = np.floor((img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8) * (levels - 1)).astype(np.int32)

    offsets = [(0, 1), (1, 0), (1, 1), (1, -1),  # d=1, 0°/90°/45°/135°
               (0, 2), (2, 0), (2, 2), (2, -2)]   # d=2
    feats = []

    for dy, dx in offsets:
        glcm = _glcm_matrix(img_q, levels, dy, dx)
        if glcm.sum() == 0:
            feats.extend([0.0] * 5)
            continue

        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        diff = np.abs(ii.astype(float) - jj.astype(float))

        contrast = float((glcm * (diff ** 2)).sum())
        dissimilarity = float((glcm * diff).sum())
        homogeneity = float((glcm / (1.0 + diff)).sum())
        energy = float((glcm ** 2).sum())
        # 相关性
        mu_i = (glcm.sum(axis=1) * np.arange(levels)).sum()
        mu_j = (glcm.sum(axis=0) * np.arange(levels)).sum()
        si = np.sqrt(((glcm.sum(axis=1) * (np.arange(levels) - mu_i) ** 2)).sum())
        sj = np.sqrt(((glcm.sum(axis=0) * (np.arange(levels) - mu_j) ** 2)).sum())
        if si < 1e-8 or sj < 1e-8:
            correlation = 0.0
        else:
            correlation = float(((glcm * (ii - mu_i) * (jj - mu_j)).sum()) / (si * sj + 1e-8))

        feats.extend([contrast, dissimilarity, homogeneity, energy, correlation])

    return feats


def _lbp_histogram(img: np.ndarray) -> list:
    """Uniform LBP 直方图 (r=1, 8 neighbors → 10 bins)"""
    h, w = img.shape
    n_points = 8
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dy = -np.round(np.sin(angles)).astype(int)
    dx = np.round(np.cos(angles)).astype(int)

    lbp = np.zeros((h - 2, w - 2), dtype=np.int32)
    center = img[1:h - 1, 1:w - 1]

    for k in range(n_points):
        neighbor = img[1 + dy[k]:h - 1 + dy[k], 1 + dx[k]:w - 1 + dx[k]]
        lbp += ((neighbor >= center).astype(np.int32)) << k

    # 统计 uniform patterns (二进制跳变次数 ≤ 2)
    transitions = np.zeros_like(lbp, dtype=np.int32)
    for k in range(n_points):
        bit_k = (lbp >> k) & 1
        bit_next = (lbp >> ((k + 1) % n_points)) & 1
        transitions += (bit_k ^ bit_next)

    is_uniform = transitions <= 2
    uniform_codes = lbp[is_uniform]
    # 对 uniform codes 重新映射到 [0, 58] (最多 59 类)
    # 简化: 统计各 bitcount，保留 58 个 bin + 1 个非 uniform bin
    hist = np.zeros(10, dtype=np.float64)
    if uniform_codes.size > 0:
        # 按方向数粗略分组: bins 1-8 + non-uniform + zero
        popcount = np.zeros(uniform_codes.size, dtype=np.int32)
        for k in range(n_points):
            popcount += ((uniform_codes >> k) & 1).astype(np.int32)
        for b in range(1, 9):
            hist[b - 1] = (popcount == b).sum()
    hist[8] = (~is_uniform).sum()
    hist[9] = (lbp == 0).sum()
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


def _gradient_orientation_histogram(img: np.ndarray, bins: int = 9) -> list:
    """简化 HOG: 梯度方向直方图 (9 bins, 0~180°)"""
    gx = sobel(img.astype(float), axis=1)
    gy = sobel(img.astype(float), axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    orient = (np.arctan2(gy, gx) % np.pi) / np.pi  # [0, 1)
    hist, _ = np.histogram(orient, bins=bins, range=(0, 1), weights=mag)
    hist /= (hist.sum() + 1e-8)
    return hist.tolist()


def _entropy(prob: np.ndarray) -> float:
    """香农熵"""
    p = prob[prob > 0]
    return float(-(p * np.log2(p)).sum())


def _hu_moments(img: np.ndarray) -> list:
    """7 个 Hu 不变矩 (在二值化图上计算，捕捉形状)"""
    binary = (img > np.percentile(img, 50)).astype(np.float64)
    total = binary.sum()
    if total < 1:
        return [0.0] * 7

    h, w = binary.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xm = (xx * binary).sum() / total
    ym = (yy * binary).sum() / total

    def _mu(p, q):
        dx = (xx - xm) ** p
        dy = (yy - ym) ** q
        return (dx * dy * binary).sum() / total

    n02 = _mu(0, 2)
    n03 = _mu(0, 3)
    n11 = _mu(1, 1)
    n12 = _mu(1, 2)
    n20 = _mu(2, 0)
    n21 = _mu(2, 1)
    n30 = _mu(3, 0)

    hu = []
    hu.append(n20 + n02)
    hu.append((n20 - n02) ** 2 + 4 * n11 ** 2)
    hu.append((n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2)
    hu.append((n30 + n12) ** 2 + (n21 + n03) ** 2)
    hu.append((n30 - 3 * n12) * (n30 + n12) *
               ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
               (3 * n21 - n03) * (n21 + n03) *
               (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    hu.append((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) +
               4 * n11 * (n30 + n12) * (n21 + n03))
    hu.append((3 * n21 - n03) * (n30 + n12) *
               ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
               (n30 - 3 * n12) * (n21 + n03) *
               (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))

    out = []
    for val in hu:
        out.append(np.sign(val) * np.log10(abs(val) + 1e-8))
    return out


def _fourier_features(img: np.ndarray, n_bands: int = 8) -> list:
    """傅里叶径向频谱能量 (按频带划分)"""
    f = np.fft.fftshift(np.fft.fft2(img.astype(float)))
    ps = np.abs(f) ** 2
    h, w = ps.shape
    center_y, center_x = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    rr = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    max_r = min(center_y, center_x)

    feats = []
    for b in range(1, n_bands + 1):
        r_low = max_r * (b - 1) / n_bands
        r_high = max_r * b / n_bands
        mask = (rr >= r_low) & (rr < r_high)
        band_energy = ps[mask].sum()
        feats.append(float(np.log10(band_energy + 1e-8)))

    total_energy = ps.sum()
    feats.append(float(np.log10(total_energy + 1e-8)))
    return feats


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    输入: (32, 32) 灰度 numpy 数组, dtype=uint8 或 float
    输出: 一维特征向量 (~400 维)
    """
    h, w = img.shape
    img_f = img.astype(np.float64)
    features = []

    # ── 1. 灰度直方图 (32 bins, 细粒度) ──
    hist, _ = np.histogram(img, bins=32, range=(0, 255))
    hist = hist.astype(np.float64) / img.size
    features.extend(hist.tolist())

    # ── 2. 累积直方图特征 ──
    cumsum = np.cumsum(hist)
    for pct in [0.1, 0.25, 0.5, 0.75, 0.9]:
        idx = np.searchsorted(cumsum, pct)
        features.append(float(idx) / 32.0)

    # ── 3. 全局统计 + 高阶矩 ──
    mu = float(np.mean(img_f))
    sigma = float(np.std(img_f))
    features.append(mu)
    features.append(sigma)
    features.append(float(np.min(img_f)))
    features.append(float(np.max(img_f)))
    features.append(float(np.median(img_f)))
    features.append(float(np.percentile(img_f, 25)))
    features.append(float(np.percentile(img_f, 75)))
    # 偏度 & 峰度
    cent = img_f - mu
    m3 = np.mean(cent ** 3)
    m4 = np.mean(cent ** 4)
    features.append(float(m3 / (sigma ** 3 + 1e-8)))  # skewness
    features.append(float(m4 / (sigma ** 4 + 1e-8)))  # kurtosis

    # ── 4. 对比度指标 ──
    # RMS 对比度
    rms_contrast = sigma / (mu + 1e-8)
    features.append(float(rms_contrast))
    # Michelson 对比度 (用于周期性图案)
    michelson = (float(np.max(img_f)) - float(np.min(img_f))) / \
                (float(np.max(img_f)) + float(np.min(img_f)) + 1e-8)
    features.append(float(michelson))
    # 局部对比度 (相邻像素差)
    diff_h = np.abs(img_f[:, 1:] - img_f[:, :-1])
    diff_v = np.abs(img_f[1:, :] - img_f[:-1, :])
    features.append(float(np.mean(diff_h)))
    features.append(float(np.std(diff_h)))
    features.append(float(np.mean(diff_v)))
    features.append(float(np.std(diff_v)))

    # ── 5. 熵 ──
    features.append(_entropy(hist + 1e-8))
    # 局部熵 (4x4 patches)
    psz = h // 4
    for i in range(4):
        for j in range(4):
            patch = img_f[i * psz:(i + 1) * psz, j * psz:(j + 1) * psz]
            phist, _ = np.histogram(patch, bins=16, range=(0, 255))
            phist = phist.astype(np.float64) / patch.size + 1e-8
            features.append(_entropy(phist))

    # ── 6. 分区统计 (3x3 = 9 区, 更细粒度) ──
    gsz = h // 3
    for i in range(3):
        for j in range(3):
            patch = img_f[i * gsz:(i + 1) * gsz, j * gsz:(j + 1) * gsz]
            pm = float(np.mean(patch))
            ps = float(np.std(patch))
            features.append(pm)
            features.append(ps)
            features.append(pm / (ps + 1e-8))  # 信噪比

    # ── 7. 行/列投影统计 ──
    row_means = np.mean(img_f, axis=1)
    col_means = np.mean(img_f, axis=0)
    features.append(float(np.mean(row_means)))
    features.append(float(np.std(row_means)))
    features.append(float(np.mean(col_means)))
    features.append(float(np.std(col_means)))
    # 行/列自相关粗略统计
    features.append(float(np.corrcoef(row_means[:-1], row_means[1:])[0, 1]
                         if len(row_means) > 1 else 0))
    features.append(float(np.corrcoef(col_means[:-1], col_means[1:])[0, 1]
                         if len(col_means) > 1 else 0))

    # ── 8. 重心 ──
    total = img_f.sum()
    if total > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cx = float((xx * img_f).sum() / total) / w
        cy = float((yy * img_f).sum() / total) / h
    else:
        cx = cy = 0.5
    features.append(cx)
    features.append(cy)

    # ── 9. 边缘 / 梯度特征 ──
    gx = sobel(img_f, axis=1)
    gy = sobel(img_f, axis=0)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    gdir = np.arctan2(gy, gx)  # [-π, π]
    features.append(float(np.mean(gmag)))
    features.append(float(np.std(gmag)))
    features.append(float(np.max(gmag)))
    features.append(float(np.mean(np.abs(gx))))
    features.append(float(np.mean(np.abs(gy))))
    # 梯度方向统计
    features.append(float(np.mean(gdir)))
    features.append(float(np.std(gdir)))

    # ── 10. 梯度方向直方图 (HOG 9 bins) ──
    features.extend(_gradient_orientation_histogram(img, bins=9))

    # ── 11. GLCM 纹理特征 (8 方向/距离 × 5 Haralick = 40 维) ──
    features.extend(_glcm_features(img, levels=8))

    # ── 12. LBP 直方图 (10 bins) ──
    features.extend(_lbp_histogram(img))

    # ── 13. Hu 不变矩 (7 维) ──
    features.extend(_hu_moments(img))

    # ── 14. 傅里叶频谱 (9 维) ──
    features.extend(_fourier_features(img, n_bands=8))

    # ── 15. 降采样像素值 (8x8 → 64 维) ──
    small = np.array(Image.fromarray(img).resize((8, 8), Image.BILINEAR)).astype(np.float64)
    features.extend(small.flatten().tolist())

    # ── 16. 二值化形态特征 ──
    for thresh_pct in [30, 50, 70]:
        binary = (img_f > np.percentile(img_f, thresh_pct)).astype(np.float64)
        bw_ratio = binary.sum() / binary.size  # 黑白比
        row_sums = binary.sum(axis=1)
        col_sums = binary.sum(axis=0)
        features.append(float(bw_ratio))
        features.append(float(np.sum(row_sums > 0)))  # 非零行数
        features.append(float(np.sum(col_sums > 0)))  # 非零列数
        features.append(float(row_sums.max()))
        features.append(float(col_sums.max()))
        # 二值图的惯性矩
        y_idxs, x_idxs = np.where(binary > 0)
        if len(y_idxs) > 1:
            features.append(float(np.std(y_idxs)))
            features.append(float(np.std(x_idxs)))
        else:
            features.append(0.0)
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def build_feature_names():
    """返回特征名称列表 (方便查看特征重要性)"""
    names = []
    names += [f"hist_{i}" for i in range(32)]
    names += ["cumpct_10", "cumpct_25", "cumpct_50", "cumpct_75", "cumpct_90"]
    names += ["mean", "std", "min", "max", "median", "p25", "p75", "skewness", "kurtosis"]
    names += ["rms_contrast", "michelson_contrast",
              "diff_h_mean", "diff_h_std", "diff_v_mean", "diff_v_std"]
    names += ["global_entropy"]
    names += [f"entropy_p{i}_{j}" for i in range(4) for j in range(4)]
    names += [f"grid3_{i}_{j}_{stat}" for i in range(3) for j in range(3)
              for stat in ["mean", "std", "snr"]]
    names += ["row_mean_of_means", "row_std_of_means",
              "col_mean_of_means", "col_std_of_means",
              "row_autocorr", "col_autocorr"]
    names += ["cx", "cy"]
    names += ["gradmag_mean", "gradmag_std", "gradmag_max",
              "gradx_mean", "grady_mean", "graddir_mean", "graddir_std"]
    names += [f"hog_{i}" for i in range(9)]
    for offset in ["d1_0", "d1_90", "d1_45", "d1_135", "d2_0", "d2_90", "d2_45", "d2_135"]:
        names += [f"glcm_{offset}_{f}" for f in
                  ["contrast", "dissim", "homo", "energy", "corr"]]
    names += [f"lbp_{i}" for i in range(10)]
    names += [f"hu_{i}" for i in range(7)]
    names += [f"fourier_band_{i}" for i in range(8)] + ["fourier_total"]
    names += [f"down_{i}" for i in range(64)]
    for thr in [30, 50, 70]:
        names += [f"bw_{thr}_ratio", f"bw_{thr}_nrows", f"bw_{thr}_ncols",
                  f"bw_{thr}_maxrow", f"bw_{thr}_maxcol",
                  f"bw_{thr}_ystd", f"bw_{thr}_xstd"]
    return names


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_dataset(data_dir=DATA_DIR):
    """从 data/ 目录加载所有图像，子文件夹名为类别标签"""
    if not os.path.isdir(data_dir):
        print(f"[错误] 数据目录不存在: {data_dir}")
        sys.exit(1)

    X, y = [], []
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not class_names:
        print(f"[错误] {data_dir}/ 下未找到任何类别文件夹")
        sys.exit(1)

    label_map = {name: idx for idx, name in enumerate(class_names)}
    print(f"[数据] 类别数: {len(class_names)}  →  {class_names}")

    for cls_name in class_names:
        cls_dir = os.path.join(data_dir, cls_name)
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(cls_dir, ext)))
        if not files:
            print(f"[警告] 类别 '{cls_name}' 下未找到图像文件")
            continue

        for fp in files:
            try:
                img = Image.open(fp).convert("L").resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.uint8)
                feats = extract_features(arr)
                X.append(feats)
                y.append(label_map[cls_name])
            except Exception as e:
                print(f"[跳过] {fp}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[数据] 总样本: {len(X)}, 特征维度: {X.shape[1]}")
    print(f"[数据] 类别分布: {dict(zip(class_names, Counter(y).values()))}")
    return X, y, class_names, label_map


# ═══════════════════════════════════════════════════════════════
#  训练
# ═══════════════════════════════════════════════════════════════

def train(args):
    X, y, class_names, label_map = load_dataset(args.data_dir)

    num_class = len(class_names)
    params = {
        "objective": "multi:softmax" if not args.prob else "multi:softprob",
        "num_class": num_class,
        "max_depth": args.depth,
        "learning_rate": args.lr,
        "n_estimators": args.rounds,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "reg_alpha": args.alpha,
        "reg_lambda": args.lambd,
        "eval_metric": "mlogloss",
        "verbosity": 1,
        "n_jobs": -1,
        "random_state": args.seed,
    }

    if args.split:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=args.split, random_state=args.seed, stratify=y
        )
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50 if args.verbose else False,
        )
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"\n[验证] Acc: {acc:.4f}")
        print(classification_report(y_val, y_pred, target_names=class_names))
    else:
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, verbose=50 if args.verbose else False)

    # ── CV 评估 ──
    if args.cv:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        scores = cross_val_score(
            xgb.XGBClassifier(**params), X, y,
            cv=skf, scoring="accuracy", n_jobs=-1,
        )
        print(f"[CV] {args.cv}-折交叉验证 Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    save_model(model, class_names, label_map)
    return model


def save_model(model, class_names, label_map):
    bundle = {
        "model": model,
        "class_names": class_names,
        "label_map": label_map,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[保存] 模型已保存至: {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 模型文件不存在: {MODEL_PATH}，请先训练")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["class_names"], bundle["label_map"]


# ═══════════════════════════════════════════════════════════════
#  推理
# ═══════════════════════════════════════════════════════════════

def predict(args):
    model, class_names, _ = load_model()
    img = Image.open(args.image).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.uint8)
    feats = extract_features(arr).reshape(1, -1)

    pred = model.predict(feats)[0]
    proba = model.predict_proba(feats)[0]

    print(f"[预测] 类别: {class_names[pred]}  (id={pred})")
    print(f"[概率] ", end="")
    order = np.argsort(proba)[::-1]
    for idx in order[:5]:
        print(f"{class_names[idx]}:{proba[idx]:.4f}", end="  ")
    print()

    if args.topk:
        for idx in order[:args.topk]:
            print(f"  top-{list(order).index(idx)+1}: {class_names[idx]} ({proba[idx]:.4f})")


# ═══════════════════════════════════════════════════════════════
#  评估
# ═══════════════════════════════════════════════════════════════

def evaluate(args):
    X, y, class_names, label_map = load_dataset(args.data_dir)
    model, _, _ = load_model()
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"[评估] 整体准确率: {acc:.4f}")
    print(classification_report(y, y_pred, target_names=class_names))
    cm = confusion_matrix(y, y_pred)
    print("[混淆矩阵]")
    print("           " + "".join(f"{n:>8s}" for n in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10s} " + "".join(f"{v:8d}" for v in row))


# ═══════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="XGBoost 图像分类器")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--data_dir", default=DATA_DIR)
    p_train.add_argument("--depth", type=int, default=6)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--rounds", type=int, default=300)
    p_train.add_argument("--subsample", type=float, default=0.8)
    p_train.add_argument("--colsample", type=float, default=0.8)
    p_train.add_argument("--alpha", type=float, default=0.1)
    p_train.add_argument("--lambd", type=float, default=1.0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--split", type=float, default=0.2,
                          help="验证集比例 (0=不划分)")
    p_train.add_argument("--cv", type=int, default=0,
                          help="交叉验证折数 (0=不做CV)")
    p_train.add_argument("--prob", action="store_true",
                          help="使用 softprob 目标")
    p_train.add_argument("--verbose", action="store_true", default=True)

    # predict
    p_pred = sub.add_parser("predict", help="单张推理")
    p_pred.add_argument("image", help="图像路径")
    p_pred.add_argument("--topk", type=int, default=5,
                         help="显示前 K 个概率")

    # eval
    p_eval = sub.add_parser("eval", help="在数据集上评估模型")
    p_eval.add_argument("--data_dir", default=DATA_DIR)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "predict":
        predict(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
