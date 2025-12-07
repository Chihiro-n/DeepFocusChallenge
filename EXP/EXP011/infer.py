"""
EXP011: FOV-Normalized Features

FOV（Field of View）によるスケール差を補正する。

問題:
- FOV 2um: 1ピクセル = 2/1024 um
- FOV 4um: 1ピクセル = 4/1024 um（2倍粗い）
- 同じ物理的エッジでもFOV 4の画像はSobel値が半分になる

解決策:
- pixel_size = FOV / ImageSize を計算
- 勾配系の特徴量を pixel_size で正規化（物理単位に変換）
- これによりFOVに依存しない特徴量を得る

使用方法:
```
python EXP/EXP011/infer.py
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
OUTPUT_DIR = Path("EXP/EXP011/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 特徴量抽出（FOV正規化版）
# ============================================

def extract_blur_features(image_path: str, fov: float, image_size: int) -> dict:
    """
    画像からボケ検出用の特徴量を抽出（FOV正規化版）

    Args:
        image_path: 画像パス
        fov: Field of View (um)
        image_size: 画像サイズ (pixels)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    rows, cols = img.shape
    features = {}
    eps = 1e-8

    # ピクセルサイズ（um/pixel）
    pixel_size = fov / image_size
    # 基準ピクセルサイズ（FOV=2, ImageSize=1024）
    base_pixel_size = 2.0 / 1024
    # スケール係数（FOV=2を基準として正規化）
    scale_factor = pixel_size / base_pixel_size

    # FOV情報を特徴量として追加
    features['fov'] = float(fov)
    features['pixel_size'] = float(pixel_size)
    features['scale_factor'] = float(scale_factor)

    # ===============================
    # Part 1: 生の特徴量（EXP003ベース）
    # ===============================

    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_var = float(laplacian.var())
    laplacian_std = float(laplacian.std())
    laplacian_mean_abs = float(np.abs(laplacian).mean())

    features['laplacian_var'] = laplacian_var
    features['laplacian_std'] = laplacian_std
    features['laplacian_mean_abs'] = laplacian_mean_abs

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

    sobel_mean = float(sobel_mag.mean())
    sobel_std = float(sobel_mag.std())
    sobel_max = float(sobel_mag.max())
    sobel_p95 = float(np.percentile(sobel_mag, 95))
    sobel_p99 = float(np.percentile(sobel_mag, 99))

    features['sobel_mean'] = sobel_mean
    features['sobel_std'] = sobel_std
    features['sobel_max'] = sobel_max
    features['sobel_p95'] = sobel_p95
    features['sobel_p99'] = sobel_p99

    # Scharr
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    scharr_mean = float(scharr_mag.mean())
    scharr_std = float(scharr_mag.std())

    features['scharr_mean'] = scharr_mean
    features['scharr_std'] = scharr_std

    # Canny
    for thresh_low, thresh_high in [(50, 150), (100, 200), (30, 100)]:
        edges = cv2.Canny(img, thresh_low, thresh_high)
        features[f'canny_density_{thresh_low}_{thresh_high}'] = float(edges.mean() / 255)

    # Local contrast
    for kernel_size in [3, 5, 7, 11]:
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        lc_mean = float(local_std.mean())
        lc_std = float(local_std.std())

        features[f'local_contrast_mean_k{kernel_size}'] = lc_mean
        features[f'local_contrast_std_k{kernel_size}'] = lc_std

    # 画像統計量
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())
    features['img_p5'] = float(np.percentile(img, 5))
    features['img_p95'] = float(np.percentile(img, 95))

    # 隣接ピクセル差分
    diff_h = np.abs(img_float[:, 1:] - img_float[:, :-1])
    diff_v = np.abs(img_float[1:, :] - img_float[:-1, :])

    diff_h_mean = float(diff_h.mean())
    diff_v_mean = float(diff_v.mean())
    diff_h_std = float(diff_h.std())
    diff_v_std = float(diff_v.std())

    features['neighbor_diff_h_mean'] = diff_h_mean
    features['neighbor_diff_v_mean'] = diff_v_mean
    features['neighbor_diff_h_std'] = diff_h_std
    features['neighbor_diff_v_std'] = diff_v_std

    # Multi-scale LoG
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        features[f'log_var_s{sigma}'] = float(log.var())
        features[f'log_max_s{sigma}'] = float(np.abs(log).max())

    # Tenengrad
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    tenengrad_mean = float(tenengrad.mean())
    tenengrad_sum = float(tenengrad.sum()) / (rows * cols)

    features['tenengrad_mean'] = tenengrad_mean
    features['tenengrad_sum'] = tenengrad_sum

    # CPBD-like (JNB)
    edges = cv2.Canny(img, 50, 150)
    edge_points = np.where(edges > 0)
    if len(edge_points[0]) > 100:
        edge_gradients = sobel_mag[edge_points]
        features['edge_gradient_mean'] = float(edge_gradients.mean())
        features['edge_gradient_std'] = float(edge_gradients.std())
        features['edge_gradient_p25'] = float(np.percentile(edge_gradients, 25))
        features['edge_gradient_p75'] = float(np.percentile(edge_gradients, 75))
        for thresh in [10, 20, 30, 50]:
            sharp_ratio = np.sum(edge_gradients > thresh) / len(edge_gradients)
            features[f'jnb_sharp_ratio_t{thresh}'] = float(sharp_ratio)
    else:
        features['edge_gradient_mean'] = 0.0
        features['edge_gradient_std'] = 0.0
        features['edge_gradient_p25'] = 0.0
        features['edge_gradient_p75'] = 0.0
        for thresh in [10, 20, 30, 50]:
            features[f'jnb_sharp_ratio_t{thresh}'] = 0.0

    # ESF width
    sobel_binary = (sobel_mag > sobel_mag.mean()).astype(np.uint8)
    h_widths = []
    for row in range(0, rows, 10):
        line = sobel_binary[row, :]
        changes = np.diff(np.concatenate([[0], line, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if len(starts) > 0 and len(ends) > 0:
            widths = ends - starts
            h_widths.extend(widths)
    if h_widths:
        esf_width_mean = float(np.mean(h_widths))
        esf_width_std = float(np.std(h_widths))
        features['esf_width_h_mean'] = esf_width_mean
        features['esf_width_h_std'] = esf_width_std
    else:
        features['esf_width_h_mean'] = 0.0
        features['esf_width_h_std'] = 0.0

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
    # Part 2: FOV正規化特徴量
    # ===============================
    # 勾配系の特徴量をscale_factorで正規化
    # FOV=2を基準として、FOV=4の場合は2倍にスケーリング
    # （FOV=4は物理的に同じエッジでも勾配が半分になるため）

    # Laplacian正規化（二階微分なのでscale_factor^2）
    features['laplacian_var_norm'] = laplacian_var * (scale_factor ** 2)
    features['laplacian_std_norm'] = laplacian_std * scale_factor
    features['laplacian_mean_abs_norm'] = laplacian_mean_abs * scale_factor

    # Sobel正規化（一階微分なのでscale_factor）
    features['sobel_mean_norm'] = sobel_mean * scale_factor
    features['sobel_std_norm'] = sobel_std * scale_factor
    features['sobel_max_norm'] = sobel_max * scale_factor
    features['sobel_p95_norm'] = sobel_p95 * scale_factor
    features['sobel_p99_norm'] = sobel_p99 * scale_factor

    # Scharr正規化
    features['scharr_mean_norm'] = scharr_mean * scale_factor
    features['scharr_std_norm'] = scharr_std * scale_factor

    # Local contrast正規化
    for kernel_size in [3, 5, 7, 11]:
        features[f'local_contrast_mean_k{kernel_size}_norm'] = features[f'local_contrast_mean_k{kernel_size}'] * scale_factor
        features[f'local_contrast_std_k{kernel_size}_norm'] = features[f'local_contrast_std_k{kernel_size}'] * scale_factor

    # 隣接差分正規化
    features['neighbor_diff_h_mean_norm'] = diff_h_mean * scale_factor
    features['neighbor_diff_v_mean_norm'] = diff_v_mean * scale_factor
    features['neighbor_diff_h_std_norm'] = diff_h_std * scale_factor
    features['neighbor_diff_v_std_norm'] = diff_v_std * scale_factor

    # Tenengrad正規化（二乗なのでscale_factor^2）
    features['tenengrad_mean_norm'] = tenengrad_mean * (scale_factor ** 2)
    features['tenengrad_sum_norm'] = tenengrad_sum * (scale_factor ** 2)

    # Edge gradient正規化
    features['edge_gradient_mean_norm'] = features['edge_gradient_mean'] * scale_factor
    features['edge_gradient_std_norm'] = features['edge_gradient_std'] * scale_factor

    # ESF width正規化（幅はscale_factorで割る = 物理単位に変換）
    features['esf_width_h_mean_norm'] = features['esf_width_h_mean'] * pixel_size  # um単位
    features['esf_width_h_std_norm'] = features['esf_width_h_std'] * pixel_size

    # LoG正規化
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        features[f'log_var_s{sigma}_norm'] = features[f'log_var_s{sigma}'] * (scale_factor ** 2)
        features[f'log_max_s{sigma}_norm'] = features[f'log_max_s{sigma}'] * scale_factor

    return features


def process_images(df: pd.DataFrame, data_dir: Path, desc: str = "Processing") -> pd.DataFrame:
    """DataFrameの全画像に対して特徴量抽出"""
    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(data_dir / rel_path)
        fov = row['FOV']
        image_size = row['ImageSize']

        try:
            features = extract_blur_features(image_path, fov, image_size)
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
    print("EXP011: FOV-Normalized Features + LightGBM")
    print("=" * 60)

    # データ読み込み
    sample_df = pd.read_csv(DATA_DIR / "sample.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nSample: {len(sample_df)} images")
    print(f"Test: {len(test_df)} images")

    # FOV分布を表示
    print("\nSample FOV distribution:")
    print(sample_df['FOV'].value_counts().sort_index())
    print("\nTest FOV distribution:")
    print(test_df['FOV'].value_counts().sort_index())

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

    # 正規化特徴量の数をカウント
    norm_cols = [c for c in feature_cols if '_norm' in c]
    print(f"Normalized features: {len(norm_cols)}")

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
        fov_val = sample_features[mask]['fov'].iloc[0]
        print(f"  Pattern {pattern} (FOV={fov_val}): {pattern_rmse:.4f}")

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

    print("\nTop 20 Feature Importance:")
    print(importance.head(20).to_string(index=False))

    # 正規化特徴量の重要度
    norm_importance = importance[importance['feature'].str.contains('_norm')]
    print("\nTop 10 Normalized Features:")
    print(norm_importance.head(10).to_string(index=False))

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

    # FOV別の予測分布
    print("\nPrediction by FOV:")
    test_with_pred = test_df.copy()
    test_with_pred['pred'] = y_pred_test
    for fov in sorted(test_with_pred['FOV'].unique()):
        fov_preds = test_with_pred[test_with_pred['FOV'] == fov]['pred']
        print(f"  FOV={fov}: mean={fov_preds.mean():.2f}, std={fov_preds.std():.2f}, n={len(fov_preds)}")

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
