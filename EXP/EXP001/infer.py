"""
EXP001: LightGBM による defocus 推定

EXP000のRidge回帰をLightGBMに置き換え、非線形関係を捉える。
特徴量も追加。

使用方法:
```
python EXP/EXP001/infer.py
```
"""

import numpy as np
import pandas as pd
import cv2
import yaml
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP001/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 特徴量抽出
# ============================================

def extract_blur_features(image_path: str) -> dict:
    """画像からボケ検出用の特徴量を抽出（拡張版）"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    features = {}

    # 1. Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_std'] = float(laplacian.std())
    features['laplacian_mean_abs'] = float(np.abs(laplacian).mean())

    # 2. FFT
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    max_dist = np.sqrt(crow ** 2 + ccol ** 2)

    # 複数の周波数帯域
    for low_r, high_r, name in [(0, 0.1, 'low'), (0.1, 0.3, 'mid'), (0.3, 0.5, 'high'), (0.5, 1.0, 'vhigh')]:
        low_thresh = max_dist * low_r
        high_thresh = max_dist * high_r
        mask = (dist_from_center > low_thresh) & (dist_from_center <= high_thresh)
        energy = np.sum((magnitude * mask) ** 2)
        total_energy = np.sum(magnitude ** 2)
        features[f'fft_{name}_ratio'] = float(energy / total_energy) if total_energy > 0 else 0

    # 3. Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    features['sobel_mean'] = float(sobel_mag.mean())
    features['sobel_std'] = float(sobel_mag.std())
    features['sobel_max'] = float(sobel_mag.max())
    features['sobel_p95'] = float(np.percentile(sobel_mag, 95))
    features['sobel_p99'] = float(np.percentile(sobel_mag, 99))

    # 4. Scharr
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    features['scharr_mean'] = float(scharr_mag.mean())
    features['scharr_std'] = float(scharr_mag.std())

    # 5. Canny
    for thresh_low, thresh_high in [(50, 150), (100, 200), (30, 100)]:
        edges = cv2.Canny(img, thresh_low, thresh_high)
        features[f'canny_density_{thresh_low}_{thresh_high}'] = float(edges.mean() / 255)

    # 6. Local contrast（複数カーネルサイズ）
    for kernel_size in [3, 5, 7, 11]:
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        features[f'local_contrast_mean_k{kernel_size}'] = float(local_std.mean())
        features[f'local_contrast_std_k{kernel_size}'] = float(local_std.std())

    # 7. 画像統計量
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())
    features['img_p5'] = float(np.percentile(img, 5))
    features['img_p95'] = float(np.percentile(img, 95))

    # 8. テクスチャ特徴量（GLCM代替としてLBP的な局所パターン）
    # 隣接ピクセルとの差分統計
    diff_h = np.abs(img_float[:, 1:] - img_float[:, :-1])
    diff_v = np.abs(img_float[1:, :] - img_float[:-1, :])
    features['neighbor_diff_h_mean'] = float(diff_h.mean())
    features['neighbor_diff_v_mean'] = float(diff_v.mean())
    features['neighbor_diff_h_std'] = float(diff_h.std())
    features['neighbor_diff_v_std'] = float(diff_v.std())

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
    print("EXP001: LightGBM-based Defocus Estimation")
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

    # 学習データ
    X_train = sample_features[feature_cols].values
    y_train = sample_features['abs_focus'].values

    print(f"\n[3/4] Training LightGBM model...")

    # LightGBMパラメータ
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

    # Cross-validation for evaluation
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

    print("\nTop 10 Feature Importance:")
    print(importance.head(10).to_string(index=False))

    # テストデータ予測
    print(f"\n[4/4] Predicting test data...")
    X_test = test_features[feature_cols].values
    y_pred_test = final_model.predict(X_test)

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

    # 重要度保存
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)


if __name__ == '__main__':
    main()
