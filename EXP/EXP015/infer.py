"""
EXP015: Noise Injection + ノイズ耐性特徴量

仮説:
- Trainはノイズレス（クリーン）、Sample/Testはノイズあり
- このドメインシフトがボケ検出を妨害している
- Train画像にノイズを注入してドメインを揃える
- ノイズに弱い特徴量を除外し、ノイズ耐性のある特徴量のみ使用

手法:
1. Sampleのノイズレベルを推定
2. Train画像にノイズ注入して特徴量の基準値を計算
3. ノイズ耐性特徴量のみ使用:
   - 中周波数帯域（ノイズは高周波、構造は低〜中周波）
   - Autocorrelation（ランダムノイズは相関が低い）
   - 大きいカーネルの勾配（小カーネルはノイズに敏感）
4. Train基準でZ-score化（EXP002の知見）

使用方法:
```
python EXP/EXP015/infer.py
```
"""

import numpy as np
import pandas as pd
import cv2
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================
# パス設定
# ============================================
DATA_DIR = Path("input/DeepFocusChallenge_v2")
OUTPUT_DIR = Path("EXP/EXP015/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# ノイズ推定
# ============================================

def estimate_noise_level(img: np.ndarray) -> float:
    """画像のノイズレベルを推定（平坦領域の標準偏差）"""
    img_float = img.astype(np.float64)

    # Gaussianで平滑化
    smoothed = cv2.GaussianBlur(img_float, (5, 5), 0)

    # 差分の標準偏差 = ノイズ推定
    noise = img_float - smoothed

    # エッジ領域を除外（Sobelで検出）
    sobel = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    sobel_mag = np.abs(sobel)
    edge_threshold = np.percentile(sobel_mag, 70)
    flat_mask = sobel_mag < edge_threshold

    if flat_mask.sum() > 100:
        noise_std = np.std(noise[flat_mask])
    else:
        noise_std = np.std(noise)

    return noise_std


def add_noise(img: np.ndarray, noise_std: float) -> np.ndarray:
    """画像にガウシアンノイズを追加"""
    noise = np.random.normal(0, noise_std, img.shape)
    noisy_img = img.astype(np.float64) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# ============================================
# ノイズ耐性特徴量抽出
# ============================================

def extract_noise_robust_features(image_path: str, add_noise_std: float = 0) -> dict:
    """ノイズ耐性のある特徴量を抽出"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # ノイズ注入（Train画像用）
    if add_noise_std > 0:
        img = add_noise(img, add_noise_std)

    img_float = img.astype(np.float64)
    rows, cols = img.shape
    features = {}

    # ============================================
    # 1. 中周波数帯域（ノイズ耐性あり）
    # ============================================
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    max_dist = np.sqrt(crow ** 2 + ccol ** 2)
    total_energy = np.sum(magnitude ** 2)

    # 中周波数帯域を重視（ノイズは高周波、構造は低〜中周波）
    for low_r, high_r, name in [(0, 0.1, 'low'), (0.1, 0.3, 'mid'), (0.3, 0.5, 'high')]:
        low_thresh = max_dist * low_r
        high_thresh = max_dist * high_r
        mask = (dist_from_center > low_thresh) & (dist_from_center <= high_thresh)
        energy = np.sum((magnitude * mask) ** 2)
        features[f'fft_{name}_ratio'] = float(energy / total_energy) if total_energy > 0 else 0

    # 高周波は除外（ノイズに汚染されやすい）
    # features['fft_vhigh_ratio'] は使わない

    # ============================================
    # 2. Autocorrelation（ノイズ耐性あり）
    # ============================================
    # Vollathの方法: 隣接ピクセルの相関
    autocorr_h = np.mean(img_float[:, :-1] * img_float[:, 1:]) - np.mean(img_float) ** 2
    autocorr_v = np.mean(img_float[:-1, :] * img_float[1:, :]) - np.mean(img_float) ** 2
    features['autocorr_h'] = float(autocorr_h)
    features['autocorr_v'] = float(autocorr_v)
    features['autocorr_total'] = float(autocorr_h + autocorr_v)

    # 正規化版
    img_var = np.var(img_float)
    if img_var > 0:
        features['autocorr_h_norm'] = float(autocorr_h / img_var)
        features['autocorr_v_norm'] = float(autocorr_v / img_var)
    else:
        features['autocorr_h_norm'] = 0.0
        features['autocorr_v_norm'] = 0.0

    # ============================================
    # 3. 大きいカーネルの勾配（ノイズ平滑化効果）
    # ============================================
    # k=7, k=11 のみ使用（k=3は除外）
    for ksize in [7, 11]:
        # Sobelの前に平滑化
        blurred = cv2.GaussianBlur(img_float, (ksize, ksize), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        features[f'sobel_mean_k{ksize}'] = float(sobel_mag.mean())
        features[f'sobel_std_k{ksize}'] = float(sobel_mag.std())
        features[f'sobel_p95_k{ksize}'] = float(np.percentile(sobel_mag, 95))

    # ============================================
    # 4. 大きいカーネルの局所コントラスト
    # ============================================
    for kernel_size in [11, 21, 32]:
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        features[f'local_contrast_mean_k{kernel_size}'] = float(local_std.mean())
        features[f'local_contrast_std_k{kernel_size}'] = float(local_std.std())

    # ============================================
    # 5. Multi-scale LoG（大きいσのみ）
    # ============================================
    for sigma in [3.0, 5.0, 7.0]:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        features[f'log_var_s{sigma}'] = float(log.var())

    # ============================================
    # 6. Tenengrad（平滑化後）
    # ============================================
    blurred = cv2.GaussianBlur(img_float, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    features['tenengrad_mean'] = float(tenengrad.mean())

    # ============================================
    # 7. Morphological features（構造的エッジ）
    # ============================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Top-hat: 明るい構造を抽出
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    features['tophat_mean'] = float(tophat.mean())
    features['tophat_std'] = float(tophat.std())

    # Black-hat: 暗い構造を抽出
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    features['blackhat_mean'] = float(blackhat.mean())
    features['blackhat_std'] = float(blackhat.std())

    # Morphological gradient
    morph_grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    features['morph_grad_mean'] = float(morph_grad.mean())
    features['morph_grad_std'] = float(morph_grad.std())

    # ============================================
    # 8. Power spectrum slope（中周波数領域のみ）
    # ============================================
    mid_start = int(max_dist * 0.1)
    mid_end = int(max_dist * 0.4)

    radial_profile = []
    for r in range(mid_start, mid_end):
        mask = (dist_from_center >= r - 0.5) & (dist_from_center < r + 0.5)
        if mask.sum() > 0:
            radial_profile.append(np.mean(magnitude[mask] ** 2))

    if len(radial_profile) > 5:
        radial_profile = np.array(radial_profile)
        log_r = np.log(np.arange(mid_start, mid_start + len(radial_profile)))
        log_power = np.log(radial_profile + 1e-10)
        slope, _ = np.polyfit(log_r, log_power, 1)
        features['spectrum_slope_mid'] = float(slope)
    else:
        features['spectrum_slope_mid'] = 0.0

    # ============================================
    # 9. 画像統計量（正規化済み）
    # ============================================
    features['img_mean'] = float(img.mean())
    features['img_std'] = float(img.std())
    features['img_cv'] = float(img.std() / (img.mean() + 1e-6))  # 変動係数

    return features


def process_images(df: pd.DataFrame, data_dir: Path, desc: str = "Processing",
                   add_noise_std: float = 0) -> pd.DataFrame:
    """DataFrameの全画像に対して特徴量抽出"""
    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(data_dir / rel_path)

        try:
            features = extract_noise_robust_features(image_path, add_noise_std)
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
    print("EXP015: Noise Injection + ノイズ耐性特徴量")
    print("=" * 60)

    # データ読み込み
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    sample_df = pd.read_csv(DATA_DIR / "sample.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"\nTrain: {len(train_df)} images")
    print(f"Sample: {len(sample_df)} images")
    print(f"Test: {len(test_df)} images")

    # ============================================
    # Step 1: Sampleのノイズレベル推定
    # ============================================
    print("\n[1/6] Estimating noise level from Sample images...")

    noise_levels = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Noise estimation"):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(DATA_DIR / rel_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            noise_std = estimate_noise_level(img)
            noise_levels.append(noise_std)

    sample_noise_mean = np.mean(noise_levels)
    sample_noise_std = np.std(noise_levels)
    print(f"\nSample noise level: mean={sample_noise_mean:.2f}, std={sample_noise_std:.2f}")

    # Trainのノイズレベルも確認（比較用）
    train_noise_levels = []
    for idx, row in tqdm(train_df.head(100).iterrows(), total=100, desc="Train noise check"):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(DATA_DIR / rel_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            noise_std = estimate_noise_level(img)
            train_noise_levels.append(noise_std)

    train_noise_mean = np.mean(train_noise_levels)
    print(f"Train noise level (100 samples): mean={train_noise_mean:.2f}")
    print(f"Noise difference: Sample - Train = {sample_noise_mean - train_noise_mean:.2f}")

    # ============================================
    # Step 2: Train画像にノイズ注入して特徴量の基準値を計算
    # ============================================
    print("\n[2/6] Extracting Train features (with noise injection)...")

    # ノイズ注入量を決定
    noise_to_inject = max(0, sample_noise_mean - train_noise_mean)
    print(f"Noise to inject: {noise_to_inject:.2f}")

    train_features = process_images(train_df, DATA_DIR, "Train (noisy)", add_noise_std=noise_to_inject)

    # Train特徴量の統計量を計算（Z-score化用）
    feature_cols = [c for c in train_features.columns if c != 'id']
    train_stats = {}
    for col in feature_cols:
        train_stats[col] = {
            'mean': train_features[col].mean(),
            'std': train_features[col].std()
        }

    print(f"\nNumber of features: {len(feature_cols)}")

    # ============================================
    # Step 3: Sample特徴量抽出（ノイズ注入なし）
    # ============================================
    print("\n[3/6] Extracting Sample features...")
    sample_features = process_images(sample_df, DATA_DIR, "Sample", add_noise_std=0)
    sample_features = sample_features.merge(sample_df[['id', 'abs_focus', 'pattern']], on='id')

    # ============================================
    # Step 4: Test特徴量抽出
    # ============================================
    print("\n[4/6] Extracting Test features...")
    test_features = process_images(test_df, DATA_DIR, "Test", add_noise_std=0)

    # ============================================
    # Z-score化（Train基準）
    # ============================================
    print("\n[5/6] Z-score normalization (Train-based)...")

    for col in feature_cols:
        mean = train_stats[col]['mean']
        std = train_stats[col]['std']
        if std > 0:
            sample_features[f'{col}_zscore'] = (sample_features[col] - mean) / std
            test_features[f'{col}_zscore'] = (test_features[col] - mean) / std
        else:
            sample_features[f'{col}_zscore'] = 0
            test_features[f'{col}_zscore'] = 0

    # Z-score特徴量のみ使用
    zscore_cols = [c for c in sample_features.columns if c.endswith('_zscore')]
    print(f"Z-score features: {len(zscore_cols)}")

    # 特徴量保存
    sample_features.to_csv(OUTPUT_DIR / "sample_features.csv", index=False)
    test_features.to_csv(OUTPUT_DIR / "test_features.csv", index=False)

    # ============================================
    # Step 6: LightGBM学習
    # ============================================
    print("\n[6/6] Training LightGBM model...")

    X_train = sample_features[zscore_cols].values
    y_train = sample_features['abs_focus'].values
    groups = sample_features['pattern'].values

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

    # KFold CV
    print("\n--- Standard KFold CV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_kf = np.zeros(len(X_train))

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

        oof_preds_kf[val_idx] = model.predict(X_val)

    cv_rmse_kf = np.sqrt(np.mean((y_train - oof_preds_kf) ** 2))
    print(f"\nKFold CV RMSE: {cv_rmse_kf:.4f}")

    # Pattern別RMSE
    print("\nPattern-wise KFold CV RMSE:")
    pattern_rmses_kf = []
    for pattern in sorted(sample_features['pattern'].unique()):
        mask = groups == pattern
        pattern_rmse = np.sqrt(np.mean((y_train[mask] - oof_preds_kf[mask]) ** 2))
        pattern_rmses_kf.append(pattern_rmse)
        print(f"  Pattern {pattern}: {pattern_rmse:.4f}")
    macro_rmse_kf = np.mean(pattern_rmses_kf)
    print(f"  Macro RMSE: {macro_rmse_kf:.4f}")

    # GroupKFold CV
    print("\n--- GroupKFold CV (過学習検知用) ---")
    gkf = GroupKFold(n_splits=5)
    oof_preds_gkf = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
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

        oof_preds_gkf[val_idx] = model.predict(X_val)

    cv_rmse_gkf = np.sqrt(np.mean((y_train - oof_preds_gkf) ** 2))
    print(f"\nGroupKFold CV RMSE: {cv_rmse_gkf:.4f}")

    # Pattern別RMSE
    print("\nPattern-wise GroupKFold CV RMSE:")
    pattern_rmses_gkf = []
    for pattern in sorted(sample_features['pattern'].unique()):
        mask = groups == pattern
        pattern_rmse = np.sqrt(np.mean((y_train[mask] - oof_preds_gkf[mask]) ** 2))
        pattern_rmses_gkf.append(pattern_rmse)
        print(f"  Pattern {pattern}: {pattern_rmse:.4f}")
    macro_rmse_gkf = np.mean(pattern_rmses_gkf)
    print(f"  Macro RMSE: {macro_rmse_gkf:.4f}")

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
        'feature': zscore_cols,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Feature Importance:")
    print(importance.head(20).to_string(index=False))

    # テストデータ予測
    X_test = test_features[zscore_cols].values
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

    # 結果サマリー
    print("\n" + "=" * 60)
    print("CV Results Summary")
    print("=" * 60)
    print(f"\nNoise injection: {noise_to_inject:.2f}")
    print(f"KFold CV RMSE:      {cv_rmse_kf:.4f} (Macro: {macro_rmse_kf:.4f})")
    print(f"GroupKFold CV RMSE: {cv_rmse_gkf:.4f} (Macro: {macro_rmse_gkf:.4f})")
    print(f"\n過学習度合い: GroupKFold - KFold = {cv_rmse_gkf - cv_rmse_kf:.4f}")

    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(y_train, bins=20, alpha=0.7, label='Sample (true)')
    axes[0].hist(oof_preds_kf, bins=20, alpha=0.7, label='Sample (OOF pred)')
    axes[0].set_xlabel('abs_focus')
    axes[0].set_title(f'KFold: True vs OOF (RMSE={cv_rmse_kf:.2f})')
    axes[0].legend()

    axes[1].hist(y_pred_test, bins=30, alpha=0.7, color='green')
    axes[1].set_xlabel('abs_focus')
    axes[1].set_title('Test: Predicted Distribution')

    axes[2].barh(importance.head(15)['feature'], importance.head(15)['importance'])
    axes[2].set_xlabel('Importance')
    axes[2].set_title('Top 15 Feature Importance')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'prediction_distribution.png'}")
    plt.close()

    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # CV結果をCSVに保存
    cv_results = pd.DataFrame([
        {'method': 'KFold', 'cv_rmse': cv_rmse_kf, 'macro_rmse': macro_rmse_kf},
        {'method': 'GroupKFold', 'cv_rmse': cv_rmse_gkf, 'macro_rmse': macro_rmse_gkf}
    ])
    cv_results.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    # ノイズ情報を保存
    noise_info = pd.DataFrame([{
        'sample_noise_mean': sample_noise_mean,
        'sample_noise_std': sample_noise_std,
        'train_noise_mean': train_noise_mean,
        'noise_injected': noise_to_inject
    }])
    noise_info.to_csv(OUTPUT_DIR / "noise_info.csv", index=False)


if __name__ == '__main__':
    main()
