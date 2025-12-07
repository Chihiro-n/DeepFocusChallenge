"""
EXP007: Reblur-ratio Features

追加ぼかしを適用して、元画像との特徴量差分を計算する。
既にボケている画像はぼかしを加えても変化が少ない。

原理:
- シャープな画像: ぼかすと特徴量が大きく変化
- ボケた画像: 追加ぼかしでも変化が少ない

特徴量:
- F_diff = F_orig - F_blur (特徴量の減少量)
- F_ratio = F_blur / (F_orig + eps) (相対的な変化)

使用方法:
```
python EXP/EXP007/infer.py
```
"""

import numpy as np
import pandas as pd
import cv2
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP007/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 特徴量抽出
# ============================================

def extract_core_features(img: np.ndarray) -> dict:
    """コア特徴量を抽出（画像から直接）"""
    img_float = img.astype(np.float64)
    rows, cols = img.shape
    features = {}

    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_mean_abs'] = float(np.abs(laplacian).mean())

    # Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    features['sobel_mean'] = float(sobel_mag.mean())
    features['sobel_max'] = float(sobel_mag.max())
    features['sobel_p95'] = float(np.percentile(sobel_mag, 95))

    # Tenengrad
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    features['tenengrad_mean'] = float(tenengrad.mean())

    # Local contrast (k=5)
    local_mean = cv2.blur(img_float, (5, 5))
    local_sq_mean = cv2.blur(img_float ** 2, (5, 5))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    features['local_contrast_mean'] = float(local_std.mean())
    features['local_contrast_std'] = float(local_std.std())

    # FFT high frequency
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_dist = np.sqrt(crow ** 2 + ccol ** 2)
    total_energy = np.sum(magnitude ** 2)

    # High frequency ratio
    high_mask = dist_from_center > max_dist * 0.3
    high_energy = np.sum((magnitude * high_mask) ** 2)
    features['fft_high_energy'] = float(high_energy / (total_energy + 1e-8))

    # Canny edge density
    edges = cv2.Canny(img, 50, 150)
    features['edge_density'] = float(edges.mean() / 255)

    return features


def extract_reblur_features(image_path: str) -> dict:
    """Reblur-ratio特徴量を抽出"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    features = {}
    eps = 1e-8

    # 元画像の特徴量
    orig_features = extract_core_features(img)

    # 元特徴量をそのまま追加
    for key, val in orig_features.items():
        features[f'orig_{key}'] = val

    # 複数のぼかしレベルで差分・比率を計算
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        # Gaussianぼかしを適用
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        blur_features = extract_core_features(blurred)

        for key in orig_features.keys():
            orig_val = orig_features[key]
            blur_val = blur_features[key]

            # 差分: シャープな画像ほど大きい（ぼかしで大きく減少）
            features[f'diff_{key}_s{sigma}'] = float(orig_val - blur_val)

            # 比率: ボケた画像ほど1に近い（変化が少ない）
            features[f'ratio_{key}_s{sigma}'] = float(blur_val / (orig_val + eps))

            # 相対変化率: シャープな画像ほど大きい
            features[f'rel_change_{key}_s{sigma}'] = float(
                (orig_val - blur_val) / (orig_val + eps)
            )

    # 画像統計量（追加）
    img_float = img.astype(np.float64)
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())

    return features


def extract_full_features(image_path: str) -> dict:
    """全特徴量を抽出（DeepResearch + Reblur）"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    rows, cols = img.shape
    features = {}
    eps = 1e-8

    # ===============================
    # Part 1: DeepResearch特徴量 (EXP003ベース)
    # ===============================

    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_std'] = float(laplacian.std())
    features['laplacian_mean_abs'] = float(np.abs(laplacian).mean())

    # FFT
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_dist = np.sqrt(crow ** 2 + ccol ** 2)
    total_energy = np.sum(magnitude ** 2)

    for low_r, high_r, name in [(0, 0.1, 'low'), (0.1, 0.3, 'mid'), (0.3, 0.5, 'high'), (0.5, 1.0, 'vhigh')]:
        low_thresh = max_dist * low_r
        high_thresh = max_dist * high_r
        mask = (dist_from_center > low_thresh) & (dist_from_center <= high_thresh)
        energy = np.sum((magnitude * mask) ** 2)
        features[f'fft_{name}_ratio'] = float(energy / total_energy) if total_energy > 0 else 0

    # Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    features['sobel_mean'] = float(sobel_mag.mean())
    features['sobel_std'] = float(sobel_mag.std())
    features['sobel_max'] = float(sobel_mag.max())
    features['sobel_p95'] = float(np.percentile(sobel_mag, 95))
    features['sobel_p99'] = float(np.percentile(sobel_mag, 99))

    # Scharr
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
    features['scharr_mean'] = float(scharr_mag.mean())
    features['scharr_std'] = float(scharr_mag.std())

    # Canny
    for thresh_low, thresh_high in [(50, 150), (100, 200), (30, 100)]:
        edges = cv2.Canny(img, thresh_low, thresh_high)
        features[f'canny_density_{thresh_low}_{thresh_high}'] = float(edges.mean() / 255)

    # Local contrast
    for kernel_size in [3, 5, 7, 11]:
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        features[f'local_contrast_mean_k{kernel_size}'] = float(local_std.mean())
        features[f'local_contrast_std_k{kernel_size}'] = float(local_std.std())

    # 画像統計量
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())
    features['img_p5'] = float(np.percentile(img, 5))
    features['img_p95'] = float(np.percentile(img, 95))

    # 隣接ピクセル差分
    diff_h = np.abs(img_float[:, 1:] - img_float[:, :-1])
    diff_v = np.abs(img_float[1:, :] - img_float[:-1, :])
    features['neighbor_diff_h_mean'] = float(diff_h.mean())
    features['neighbor_diff_v_mean'] = float(diff_v.mean())
    features['neighbor_diff_h_std'] = float(diff_h.std())
    features['neighbor_diff_v_std'] = float(diff_v.std())

    # Multi-scale LoG
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        features[f'log_var_s{sigma}'] = float(log.var())
        features[f'log_max_s{sigma}'] = float(np.abs(log).max())

    # Tenengrad
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    features['tenengrad_mean'] = float(tenengrad.mean())
    features['tenengrad_sum'] = float(tenengrad.sum()) / (rows * cols)

    # CPBD-like (JNB)
    edges = cv2.Canny(img, 50, 150)
    edge_points = np.where(edges > 0)
    if len(edge_points[0]) > 100:
        edge_gradients = sobel_mag[edge_points]
        features['edge_gradient_mean'] = float(edge_gradients.mean())
        features['edge_gradient_std'] = float(edge_gradients.std())
        for thresh in [10, 20, 30, 50]:
            sharp_ratio = np.sum(edge_gradients > thresh) / len(edge_gradients)
            features[f'jnb_sharp_ratio_t{thresh}'] = float(sharp_ratio)
    else:
        features['edge_gradient_mean'] = 0.0
        features['edge_gradient_std'] = 0.0
        for thresh in [10, 20, 30, 50]:
            features[f'jnb_sharp_ratio_t{thresh}'] = 0.0

    # MTF-like
    if total_energy > 0:
        features['mtf_decay_mid_to_high'] = float(features['fft_mid_ratio'] / (features['fft_high_ratio'] + eps))
        features['mtf_decay_high_to_vhigh'] = float(features['fft_high_ratio'] / (features['fft_vhigh_ratio'] + eps))
        features['mtf_low_high_ratio'] = float(features['fft_low_ratio'] / (features['fft_high_ratio'] + eps))

    # Gradient histogram
    grad_hist, _ = np.histogram(sobel_mag.flatten(), bins=20, range=(0, sobel_mag.max() + 1))
    grad_hist = grad_hist / (grad_hist.sum() + eps)
    grad_hist_nonzero = grad_hist[grad_hist > 0]
    features['grad_entropy'] = float(-np.sum(grad_hist_nonzero * np.log(grad_hist_nonzero + 1e-10)))

    grad_flat = sobel_mag.flatten()
    grad_mean = grad_flat.mean()
    grad_std_val = grad_flat.std()
    if grad_std_val > 0:
        features['grad_skewness'] = float(((grad_flat - grad_mean) ** 3).mean() / (grad_std_val ** 3))
        features['grad_kurtosis'] = float(((grad_flat - grad_mean) ** 4).mean() / (grad_std_val ** 4))
    else:
        features['grad_skewness'] = 0.0
        features['grad_kurtosis'] = 0.0

    # Power spectrum slope
    radial_profile = []
    for r in range(1, min(crow, ccol)):
        mask = (dist_from_center >= r - 0.5) & (dist_from_center < r + 0.5)
        if mask.sum() > 0:
            radial_profile.append(np.mean(magnitude[mask] ** 2))

    if len(radial_profile) > 10:
        radial_profile = np.array(radial_profile)
        log_r = np.log(np.arange(1, len(radial_profile) + 1))
        log_power = np.log(radial_profile + 1e-10)
        slope, _ = np.polyfit(log_r, log_power, 1)
        features['spectrum_slope'] = float(slope)
    else:
        features['spectrum_slope'] = 0.0

    # ===============================
    # Part 2: Reblur特徴量
    # ===============================
    # コア特徴量をReblur用に抽出
    core_features = {
        'laplacian_var': features['laplacian_var'],
        'sobel_mean': features['sobel_mean'],
        'sobel_max': features['sobel_max'],
        'sobel_p95': features['sobel_p95'],
        'tenengrad_mean': features['tenengrad_mean'],
        'local_contrast_mean': features['local_contrast_mean_k5'],
        'local_contrast_std': features['local_contrast_std_k5'],
        'fft_high_ratio': features['fft_high_ratio'],
    }

    # 複数のぼかしレベルで差分・比率を計算
    for sigma in [2.0, 3.0, 5.0]:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)

        # Blurred画像のコア特徴量
        blur_laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        blur_sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        blur_sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        blur_sobel_mag = np.sqrt(blur_sobel_x ** 2 + blur_sobel_y ** 2)

        blur_float = blurred.astype(np.float64)
        blur_local_mean = cv2.blur(blur_float, (5, 5))
        blur_local_sq_mean = cv2.blur(blur_float ** 2, (5, 5))
        blur_local_std = np.sqrt(np.maximum(blur_local_sq_mean - blur_local_mean ** 2, 0))

        blur_fft = np.fft.fft2(blur_float)
        blur_fshift = np.fft.fftshift(blur_fft)
        blur_magnitude = np.abs(blur_fshift)
        blur_total = np.sum(blur_magnitude ** 2)
        high_mask = dist_from_center > max_dist * 0.3
        blur_high_energy = np.sum((blur_magnitude * high_mask) ** 2)

        blur_features = {
            'laplacian_var': float(blur_laplacian.var()),
            'sobel_mean': float(blur_sobel_mag.mean()),
            'sobel_max': float(blur_sobel_mag.max()),
            'sobel_p95': float(np.percentile(blur_sobel_mag, 95)),
            'tenengrad_mean': float((blur_sobel_x ** 2 + blur_sobel_y ** 2).mean()),
            'local_contrast_mean': float(blur_local_std.mean()),
            'local_contrast_std': float(blur_local_std.std()),
            'fft_high_ratio': float(blur_high_energy / (blur_total + eps)),
        }

        for key in core_features.keys():
            orig_val = core_features[key]
            blur_val = blur_features[key]

            # 差分: シャープな画像ほど大きい
            features[f'reblur_diff_{key}_s{sigma}'] = float(orig_val - blur_val)

            # 比率: ボケた画像ほど1に近い
            features[f'reblur_ratio_{key}_s{sigma}'] = float(blur_val / (orig_val + eps))

    return features


def process_images(df: pd.DataFrame, data_dir: Path, desc: str = "Processing") -> pd.DataFrame:
    """DataFrameの全画像に対して特徴量抽出"""
    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(data_dir / rel_path)

        try:
            features = extract_full_features(image_path)
            features['id'] = row['id']
            all_features.append(features)
        except Exception as e:
            print(f"Error: {image_path}: {e}")

    return pd.DataFrame(all_features)


# ============================================
# メイン処理
# ============================================

def main():
    print("=" * 60)
    print("EXP007: Reblur-ratio Features + LightGBM")
    print("=" * 60)

    # データ読み込み
    sample_df = pd.read_csv(DATA_DIR / "sample.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nSample: {len(sample_df)} images")
    print(f"Test: {len(test_df)} images")

    # 特徴量抽出
    print("\n[1/4] Extracting sample features...")
    sample_features = process_images(sample_df, DATA_DIR, "Sample")
    sample_features = sample_features.merge(sample_df[['id', 'abs_focus', 'pattern']], on='id')
    sample_features.to_csv(OUTPUT_DIR / "sample_features.csv", index=False)

    print("\n[2/4] Extracting test features...")
    test_features = process_images(test_df, DATA_DIR, "Test")
    test_features.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    # 特徴量カラム
    feature_cols = [c for c in sample_features.columns if c not in ['id', 'abs_focus', 'pattern']]
    print(f"\nNumber of features: {len(feature_cols)}")

    # Reblur特徴量の数をカウント
    reblur_cols = [c for c in feature_cols if 'reblur' in c]
    print(f"Reblur features: {len(reblur_cols)}")

    # 学習データ
    X_train = sample_features[feature_cols].values
    y_train = sample_features['abs_focus'].values

    print(f"\n[3/4] Training LightGBM...")

    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'max_depth': 4,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 3,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        oof_preds[val_idx] = model.predict(X_val)

    cv_rmse = np.sqrt(np.mean((y_train - oof_preds) ** 2))
    print(f"\nCV RMSE: {cv_rmse:.4f}")

    # パターン別RMSE
    print("\nPattern-wise CV RMSE:")
    for pattern in sorted(sample_features['pattern'].unique()):
        mask = sample_features['pattern'].values == pattern
        pattern_rmse = np.sqrt(np.mean((y_train[mask] - oof_preds[mask]) ** 2))
        print(f"  Pattern {pattern}: {pattern_rmse:.4f}")

    # 全データで再学習
    print("\nRetraining on full sample data...")
    train_data = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=300
    )

    # 特徴量重要度
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Feature Importance:")
    print(importance.head(15).to_string(index=False))

    # Reblur特徴量の重要度
    reblur_importance = importance[importance['feature'].str.contains('reblur')]
    print("\nTop 10 Reblur Features:")
    print(reblur_importance.head(10).to_string(index=False))

    # テストデータ予測
    print(f"\n[4/4] Predicting test data...")
    X_test = test_features[feature_cols].values
    y_pred_test = final_model.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 0, None)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_features['id'],
        'abs_focus': y_pred_test
    })
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Submission saved: {submission_path}")
    print(f"{'=' * 60}")

    # 予測分布
    print(f"\nPrediction statistics:")
    print(f"  Mean: {y_pred_test.mean():.2f}")
    print(f"  Std:  {y_pred_test.std():.2f}")
    print(f"  Min:  {y_pred_test.min():.2f}")
    print(f"  Max:  {y_pred_test.max():.2f}")

    # 可視化保存
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(y_train, bins=20, alpha=0.7, label='Sample (true)')
    axes[0].hist(oof_preds, bins=20, alpha=0.7, label='Sample (OOF pred)')
    axes[0].set_xlabel('abs_focus')
    axes[0].set_title(f'Sample: True vs OOF Predicted (RMSE={cv_rmse:.2f})')
    axes[0].legend()

    axes[1].hist(y_pred_test, bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('abs_focus')
    axes[1].set_title('Test: Predicted Distribution')

    axes[2].barh(importance.head(15)['feature'], importance.head(15)['importance'])
    axes[2].set_xlabel('Importance')
    axes[2].set_title('Top 15 Feature Importance')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'prediction_distribution.png'}")
    plt.close()

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


if __name__ == '__main__':
    main()
