"""
EXP002: Train画像を基準としたdefocus推定

Train画像（2220枚）は主にabs_focus=0（ベストフォーカス）。
これを「フォーカスが合っている状態」の基準として使用し、
Test画像がどれだけ基準から離れているかで推定する。

アプローチ:
1. Train画像の特徴量分布を計算（条件別に）
2. Test画像の特徴量と基準との差分を計算
3. Sample画像で差分とabs_focusの関係を学習
4. Test画像に適用

使用方法:
```
python EXP/EXP002/infer.py
```
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP002/outputs")
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

    # 5. Local contrast
    for kernel_size in [5, 7]:
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        features[f'local_contrast_mean_k{kernel_size}'] = float(local_std.mean())
        features[f'local_contrast_std_k{kernel_size}'] = float(local_std.std())

    # 6. 画像統計量
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


def create_condition_key(row):
    """撮影条件のキーを作成"""
    return f"{row['FOV']}_{row['HV']}_{row['VacuumMode']}"


# ============================================
# メイン処理
# ============================================

def main():
    print("=" * 60)
    print("EXP002: Train Reference-based Defocus Estimation")
    print("=" * 60)

    # データ読み込み
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    sample_df = pd.read_csv(DATA_DIR / "sample.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nTrain: {len(train_df)} images (reference for focus=0)")
    print(f"Sample: {len(sample_df)} images (with labels)")
    print(f"Test: {len(test_df)} images")

    # 条件キー作成
    train_df['condition'] = train_df.apply(create_condition_key, axis=1)
    sample_df['condition'] = sample_df.apply(create_condition_key, axis=1)
    test_df['condition'] = test_df.apply(create_condition_key, axis=1)

    print(f"\nTrain conditions: {train_df['condition'].unique()}")
    print(f"Sample conditions: {sample_df['condition'].unique()}")
    print(f"Test conditions: {test_df['condition'].unique()}")

    # 特徴量抽出
    print("\n[1/5] Extracting train features (reference)...")
    train_features = process_images(train_df, DATA_DIR, "Train")
    train_features = train_features.merge(train_df[['id', 'condition']], on='id')
    train_features.to_csv(OUTPUT_DIR / "train_features.csv", index=False)

    print("\n[2/5] Extracting sample features...")
    sample_features = process_images(sample_df, DATA_DIR, "Sample")
    sample_features = sample_features.merge(sample_df[['id', 'abs_focus', 'pattern', 'condition']], on='id')
    sample_features.to_csv(OUTPUT_DIR / "sample_features.csv", index=False)

    print("\n[3/5] Extracting test features...")
    test_features = process_images(test_df, DATA_DIR, "Test")
    test_features = test_features.merge(test_df[['id', 'condition']], on='id')
    test_features.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    # 特徴量カラム
    feature_cols = [c for c in train_features.columns if c not in ['id', 'condition']]
    print(f"\nFeatures: {feature_cols}")

    # ============================================
    # Train画像の条件別基準値を計算
    # ============================================
    print("\n[4/5] Computing reference statistics from train data...")

    # 全体の基準（条件ごとに分けない場合）
    train_mean = train_features[feature_cols].mean()
    train_std = train_features[feature_cols].std()

    print("\nTrain feature statistics (reference for focus=0):")
    for feat in feature_cols[:5]:  # 最初の5個だけ表示
        print(f"  {feat}: mean={train_mean[feat]:.4f}, std={train_std[feat]:.4f}")

    # ============================================
    # 差分特徴量の計算
    # ============================================

    def compute_diff_features(features_df, ref_mean, ref_std, feature_cols):
        """基準からの差分特徴量を計算"""
        diff_df = features_df[['id']].copy()

        for feat in feature_cols:
            # Z-score（基準からの偏差）
            diff_df[f'{feat}_zscore'] = (features_df[feat] - ref_mean[feat]) / (ref_std[feat] + 1e-8)
            # 絶対差分
            diff_df[f'{feat}_diff'] = features_df[feat] - ref_mean[feat]

        # 元の特徴量も保持
        for feat in feature_cols:
            diff_df[feat] = features_df[feat]

        return diff_df

    sample_diff = compute_diff_features(sample_features, train_mean, train_std, feature_cols)
    sample_diff = sample_diff.merge(sample_features[['id', 'abs_focus', 'pattern']], on='id')

    test_diff = compute_diff_features(test_features, train_mean, train_std, feature_cols)

    # ============================================
    # 回帰モデル学習
    # ============================================
    print("\n[5/5] Training regression model...")

    # 使用する特徴量（元特徴量 + Z-score）
    diff_feature_cols = [c for c in sample_diff.columns if c not in ['id', 'abs_focus', 'pattern']]

    # パターン内相関の高い特徴量を選択
    # EXP000の分析結果に基づき、負の相関が安定している特徴量のz-scoreを使用
    selected_features = [
        'local_contrast_std_k5', 'local_contrast_std_k7',
        'fft_mid_ratio', 'sobel_max', 'scharr_std',
        'local_contrast_std_k5_zscore', 'local_contrast_std_k7_zscore',
        'fft_mid_ratio_zscore', 'sobel_max_zscore',
        'local_contrast_std_k5_diff', 'fft_mid_ratio_diff'
    ]

    # 存在する特徴量のみ使用
    selected_features = [f for f in selected_features if f in sample_diff.columns]
    print(f"Selected features: {selected_features}")

    X_train = sample_diff[selected_features].values
    y_train = sample_diff['abs_focus'].values

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
    for pattern in sorted(sample_diff['pattern'].unique()):
        mask = sample_diff['pattern'] == pattern
        pattern_rmse = np.sqrt(np.mean((y_train[mask] - y_pred_train[mask]) ** 2))
        print(f"  Pattern {pattern}: {pattern_rmse:.4f}")

    # 係数表示
    print("\nTop feature coefficients:")
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coef': model.coef_
    }).sort_values('coef', key=abs, ascending=False)
    print(coef_df.head(10).to_string(index=False))

    # テストデータ予測
    X_test = test_diff[selected_features].values
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled)

    # 負の値をクリップ
    y_pred_test = np.clip(y_pred_test, 0, None)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_diff['id'],
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(y_train, bins=20, alpha=0.7, label='Sample (true)')
    axes[0].hist(y_pred_train, bins=20, alpha=0.7, label='Sample (pred)')
    axes[0].set_xlabel('abs_focus')
    axes[0].set_title(f'Sample: True vs Predicted (RMSE={train_rmse:.2f})')
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
