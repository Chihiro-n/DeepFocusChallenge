"""
EXP000: ボケ特徴量による defocus 推定

パターン内で一貫した相関を示す特徴量を使用:
- local_contrast_std (負の相関): 最も安定
- fft_mid_ratio (負の相関): 非常に安定
- sobel_max (負の相関): 安定

使用方法:
```
python EXP/EXP000/infer.py
```
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP000/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 特徴量抽出
# ============================================

def extract_blur_features(image_path: str) -> dict:
    """画像からボケ検出用の特徴量を抽出"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    features = {}

    # 1. Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_std'] = float(laplacian.std())

    # 2. FFT
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    max_dist = np.sqrt(crow ** 2 + ccol ** 2)
    low_thresh = max_dist * 0.1
    mid_thresh = max_dist * 0.3

    low_mask = dist_from_center <= low_thresh
    mid_mask = (dist_from_center > low_thresh) & (dist_from_center <= mid_thresh)
    high_mask = dist_from_center > mid_thresh

    total_energy = np.sum(magnitude ** 2)
    low_energy = np.sum((magnitude * low_mask) ** 2)
    mid_energy = np.sum((magnitude * mid_mask) ** 2)
    high_energy = np.sum((magnitude * high_mask) ** 2)

    features['fft_high_ratio'] = float(high_energy / total_energy) if total_energy > 0 else 0
    features['fft_mid_ratio'] = float(mid_energy / total_energy) if total_energy > 0 else 0
    features['fft_high_to_low'] = float(high_energy / low_energy) if low_energy > 0 else 0

    # 3. Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    features['sobel_mean'] = float(sobel_mag.mean())
    features['sobel_std'] = float(sobel_mag.std())
    features['sobel_max'] = float(sobel_mag.max())

    # 4. Scharr
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    features['scharr_mean'] = float(scharr_mag.mean())
    features['scharr_std'] = float(scharr_mag.std())

    # 5. Canny
    edges_low = cv2.Canny(img, 50, 150)
    features['canny_density_low'] = float(edges_low.mean() / 255)

    # 6. Local contrast
    kernel_size = 5
    local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    features['local_contrast_mean'] = float(local_std.mean())
    features['local_contrast_std'] = float(local_std.std())

    # 7. 画像統計量
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())

    return features


def process_images(df: pd.DataFrame, data_dir: Path, desc: str = "Processing") -> pd.DataFrame:
    """DataFrameの全画像に対して特徴量抽出"""
    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(data_dir / rel_path)

        try:
            features = extract_blur_features(image_path)
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
    print("EXP000: Blur Feature-based Defocus Estimation")
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

    # パターン内で一貫した相関を示す特徴量を使用
    feature_cols = ['local_contrast_std', 'fft_mid_ratio', 'sobel_max']

    print(f"\n[3/4] Training regression model...")
    print(f"Features: {feature_cols}")

    # 学習
    X_train = sample_features[feature_cols].values
    y_train = sample_features['abs_focus'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    # 学習データでの評価
    y_pred_train = model.predict(X_train_scaled)
    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    print(f"\nSample RMSE: {train_rmse:.4f}")

    # パターン別RMSE
    print("\nPattern-wise RMSE:")
    for pattern in sorted(sample_features['pattern'].unique()):
        mask = sample_features['pattern'] == pattern
        pattern_rmse = np.sqrt(np.mean((y_train[mask] - y_pred_train[mask]) ** 2))
        print(f"  Pattern {pattern}: {pattern_rmse:.4f}")

    # 係数表示
    print("\nFeature coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")

    # テストデータ予測
    print(f"\n[4/4] Predicting test data...")
    X_test = test_features[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)

    # 負の値をクリップ
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

    # 予測分布のヒストグラム保存
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_train, bins=20, alpha=0.7, label='Sample (true)')
    axes[0].hist(y_pred_train, bins=20, alpha=0.7, label='Sample (pred)')
    axes[0].set_xlabel('abs_focus')
    axes[0].set_title('Sample: True vs Predicted')
    axes[0].legend()

    axes[1].hist(y_pred_test, bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('abs_focus')
    axes[1].set_title('Test: Predicted Distribution')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'prediction_distribution.png'}")
    plt.close()


if __name__ == '__main__':
    main()
