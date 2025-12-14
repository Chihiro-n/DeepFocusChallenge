"""
EXP014: Ridge + GroupKFold (Discussion改善版)

Discussionからの改善点:
- GroupKFold: patternでグループ分割（過学習検知）
- シンプルなモデル: Ridge回帰（過学習対策）
- 周波数帯域のstd追加
- Full trainでLB確認

使用方法:
```
python EXP/EXP014/infer.py
```
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP014/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 特徴量抽出 (Discussion版 + EXP003の良い部分)
# ============================================

def extract_blur_features(image_path: str) -> dict:
    """画像からボケ検出用の特徴量を抽出"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    rows, cols = img.shape
    features = {}

    # ============================================
    # 1. ラプラシアン（シャープネス）
    # ============================================
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_mean_abs'] = float(np.mean(np.abs(laplacian)))

    # ============================================
    # 2. FFT 周波数帯域別パワー（Discussion版：mean + std）
    # ============================================
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    # Discussion版と同じ5帯域（放射状）
    bands = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 512)]
    for r_inner, r_outer in bands:
        mask = (dist_from_center >= r_inner) & (dist_from_center < r_outer)
        band_values = magnitude[mask]
        features[f'fft_band_{r_inner}_{r_outer}_mean'] = float(np.mean(band_values))
        features[f'fft_band_{r_inner}_{r_outer}_std'] = float(np.std(band_values))

    # 高周波/低周波比（Discussion版）
    high_freq = magnitude[dist_from_center > 200].mean()
    low_freq = magnitude[dist_from_center <= 50].mean()
    features['high_low_freq_ratio'] = float(high_freq / (low_freq + 1e-6))

    # ============================================
    # 3. エッジ勾配統計（Sobel）
    # ============================================
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    features['sobel_mean'] = float(np.mean(grad_mag))
    features['sobel_std'] = float(np.std(grad_mag))
    features['sobel_p95'] = float(np.percentile(grad_mag, 95))

    # ============================================
    # 4. 局所コントラスト（Discussion版：kernel=32）
    # ============================================
    kernel_size = 32
    local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
    local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    features['local_contrast_mean'] = float(np.mean(local_std))

    # ============================================
    # 5. EXP003からの追加特徴量（効果的なもの）
    # ============================================

    # Multi-scale LoG
    for sigma in [1.0, 2.0, 3.0]:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        features[f'log_var_s{sigma}'] = float(log.var())

    # Tenengrad
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    features['tenengrad_mean'] = float(tenengrad.mean())

    # Power spectrum slope
    max_dist = min(crow, ccol)
    radial_profile = []
    for r in range(1, max_dist):
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

    # 画像統計量
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
    print("EXP014: Ridge + GroupKFold (Discussion改善版)")
    print("=" * 60)

    # データ読み込み
    sample_df = pd.read_csv(DATA_DIR / "sample.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nSample: {len(sample_df)} images")
    print(f"Test: {len(test_df)} images")
    print(f"Patterns in sample: {sorted(sample_df['pattern'].unique())}")

    # 特徴量抽出
    print("\n[1/5] Extracting sample features...")
    sample_features = process_images(sample_df, DATA_DIR, "Sample")
    sample_features = sample_features.merge(sample_df[['id', 'abs_focus', 'pattern']], on='id')
    sample_features.to_csv(OUTPUT_DIR / "sample_features.csv", index=False)

    print("\n[2/5] Extracting test features...")
    test_features = process_images(test_df, DATA_DIR, "Test")
    test_features.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    # 特徴量カラム
    feature_cols = [c for c in sample_features.columns if c not in ['id', 'abs_focus', 'pattern']]
    print(f"\nNumber of features: {len(feature_cols)}")

    # 学習データ
    X = sample_features[feature_cols].values
    y = sample_features['abs_focus'].values
    groups = sample_features['pattern'].values

    # ============================================
    # [3/5] GroupKFold CV (過学習検知用)
    # ============================================
    print("\n[3/5] GroupKFold CV (pattern-based)...")

    gkf = GroupKFold(n_splits=5)

    models_to_test = {
        'Ridge_alpha1': Ridge(alpha=1.0),
        'Ridge_alpha10': Ridge(alpha=10.0),
        'Ridge_alpha100': Ridge(alpha=100.0),
        'Lasso_alpha1': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    }

    results = {}

    for model_name, model in models_to_test.items():
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # StandardScaler + Model
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx] = model_clone.predict(X_val_scaled)

        # Overall RMSE
        cv_rmse = np.sqrt(np.mean((y - oof_preds) ** 2))

        # Pattern-wise RMSE (macro average)
        pattern_rmses = []
        for pattern in sorted(sample_features['pattern'].unique()):
            mask = groups == pattern
            if mask.sum() > 0:
                pattern_rmse = np.sqrt(np.mean((y[mask] - oof_preds[mask]) ** 2))
                pattern_rmses.append(pattern_rmse)
        macro_rmse = np.mean(pattern_rmses)

        results[model_name] = {
            'cv_rmse': cv_rmse,
            'macro_rmse': macro_rmse,
            'oof_preds': oof_preds
        }

        print(f"  {model_name}: CV RMSE={cv_rmse:.4f}, Macro RMSE={macro_rmse:.4f}")

    # ベストモデル選択（Macro RMSEベース）
    best_model_name = min(results, key=lambda x: results[x]['macro_rmse'])
    print(f"\nBest model (by Macro RMSE): {best_model_name}")

    # ============================================
    # [4/5] 通常のKFold CVも参考に
    # ============================================
    print("\n[4/5] Standard KFold CV (for comparison)...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in ['Ridge_alpha10']:
        model = models_to_test[model_name]
        oof_preds = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_tr_scaled, y_tr)
            oof_preds[val_idx] = model_clone.predict(X_val_scaled)

        cv_rmse = np.sqrt(np.mean((y - oof_preds) ** 2))
        print(f"  {model_name} (KFold): CV RMSE={cv_rmse:.4f}")

    # ============================================
    # [5/5] Full train + Test prediction
    # ============================================
    print("\n[5/5] Full training and test prediction...")

    # Ridge alpha=10を使用
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_model = Ridge(alpha=10.0)
    final_model.fit(X_scaled, y)

    # Test予測
    X_test = test_features[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = final_model.predict(X_test_scaled)
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

    # ============================================
    # 結果サマリー
    # ============================================
    print("\n" + "=" * 60)
    print("CV Results Summary")
    print("=" * 60)
    print("\nGroupKFold (pattern-based) - 過学習検知:")
    for model_name, res in sorted(results.items(), key=lambda x: x[1]['macro_rmse']):
        print(f"  {model_name}: Macro RMSE={res['macro_rmse']:.4f}")

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    best_oof = results[best_model_name]['oof_preds']
    best_macro = results[best_model_name]['macro_rmse']

    axes[0].scatter(y, best_oof, alpha=0.7)
    axes[0].plot([0, 80], [0, 80], 'r--', label='y=x')
    axes[0].set_xlabel('True abs_focus')
    axes[0].set_ylabel('Predicted abs_focus')
    axes[0].set_title(f'{best_model_name} OOF (Macro RMSE={best_macro:.2f})')
    axes[0].legend()

    axes[1].hist(y, bins=20, alpha=0.7, label='True')
    axes[1].hist(best_oof, bins=20, alpha=0.7, label='OOF Pred')
    axes[1].set_xlabel('abs_focus')
    axes[1].set_title('Sample: True vs OOF Predicted')
    axes[1].legend()

    axes[2].hist(y_pred_test, bins=30, alpha=0.7, color='green')
    axes[2].set_xlabel('abs_focus')
    axes[2].set_title('Test: Predicted Distribution')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'prediction_distribution.png'}")
    plt.close()

    # CV結果をCSVに保存
    cv_results_df = pd.DataFrame([
        {'model': k, 'cv_rmse': v['cv_rmse'], 'macro_rmse': v['macro_rmse']}
        for k, v in results.items()
    ]).sort_values('macro_rmse')
    cv_results_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'cv_results.csv'}")


if __name__ == '__main__':
    main()
