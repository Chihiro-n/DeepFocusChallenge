"""
EXP012: Feature Selection from EXP003

EXP003の60特徴量から重要度の高いものだけを選択。
特徴量数を減らすことでオーバーフィットを軽減し、汎化性能を向上させる。

選択基準:
- EXP003のFeature Importanceに基づく
- Top N特徴量のみを使用

使用方法:
```
python EXP/EXP012/infer.py
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
OUTPUT_DIR = Path("EXP/EXP012/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# EXP003のFeature Importance（上位順）
# ============================================
# EXP003の結果から取得した重要度順の特徴量リスト
TOP_FEATURES_BY_IMPORTANCE = [
    'mtf_decay_high_to_vhigh',
    'jnb_sharp_ratio_t30',
    'log_var_s5.0',
    'img_std',
    'fft_low_ratio',
    'jnb_sharp_ratio_t10',
    'log_max_s5.0',
    'grad_skewness',
    'fft_mid_ratio',
    'edge_gradient_p25',
    'jnb_sharp_ratio_t20',
    'laplacian_var',
    'log_max_s3.0',
    'mtf_low_high_ratio',
    'edge_gradient_mean',
    'local_contrast_std_k5',
    'img_mean',
    'log_var_s3.0',
    'grad_entropy',
    'local_contrast_std_k7',
    'img_p95',
    'esf_width_h_mean',
    'jnb_sharp_ratio_t50',
    'sobel_p95',
    'spectrum_slope',
    'fft_high_ratio',
    'local_contrast_std_k11',
    'log_max_s2.0',
    'edge_gradient_std',
    'sobel_mean',
]


# ============================================
# 特徴量抽出（EXP003と同じ）
# ============================================

def extract_blur_features(image_path: str) -> dict:
    """画像からボケ検出用の特徴量を抽出"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    rows, cols = img.shape
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

    # 6. Local contrast
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

    # 8. 隣接ピクセル差分
    diff_h = np.abs(img_float[:, 1:] - img_float[:, :-1])
    diff_v = np.abs(img_float[1:, :] - img_float[:-1, :])
    features['neighbor_diff_h_mean'] = float(diff_h.mean())
    features['neighbor_diff_v_mean'] = float(diff_v.mean())
    features['neighbor_diff_h_std'] = float(diff_h.std())
    features['neighbor_diff_v_std'] = float(diff_v.std())

    # 9. Multi-scale LoG
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        features[f'log_var_s{sigma}'] = float(log.var())
        features[f'log_max_s{sigma}'] = float(np.abs(log).max())

    # 10. Tenengrad
    tenengrad = sobel_x ** 2 + sobel_y ** 2
    features['tenengrad_mean'] = float(tenengrad.mean())
    features['tenengrad_sum'] = float(tenengrad.sum()) / (rows * cols)

    # 11. CPBD-like (JNB)
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

    # 12. ESF width
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
        features['esf_width_h_mean'] = float(np.mean(h_widths))
        features['esf_width_h_std'] = float(np.std(h_widths))
    else:
        features['esf_width_h_mean'] = 0.0
        features['esf_width_h_std'] = 0.0

    # 13. MTF-like
    if total_energy > 0:
        features['mtf_decay_mid_to_high'] = float(features['fft_mid_ratio'] / (features['fft_high_ratio'] + 1e-8))
        features['mtf_decay_high_to_vhigh'] = float(features['fft_high_ratio'] / (features['fft_vhigh_ratio'] + 1e-8))
        features['mtf_low_high_ratio'] = float(features['fft_low_ratio'] / (features['fft_high_ratio'] + 1e-8))

    # 14. Gradient histogram
    grad_hist, _ = np.histogram(sobel_mag.flatten(), bins=20, range=(0, sobel_mag.max() + 1))
    grad_hist = grad_hist / (grad_hist.sum() + 1e-8)

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

    # 15. Power spectrum slope
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


def train_and_evaluate(X_train, y_train, feature_cols, patterns, n_features):
    """指定した数の特徴量でモデルを学習・評価"""
    # Top N特徴量を選択
    selected_features = [f for f in TOP_FEATURES_BY_IMPORTANCE[:n_features] if f in feature_cols]

    # 足りない場合は残りの特徴量から追加
    if len(selected_features) < n_features:
        remaining = [f for f in feature_cols if f not in selected_features]
        selected_features.extend(remaining[:n_features - len(selected_features)])

    X = X_train[[f for f in selected_features if f in X_train.columns]].values
    actual_n = X.shape[1]

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

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
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

    return cv_rmse, selected_features[:actual_n], oof_preds


# ============================================
# メイン処理
# ============================================

def main():
    print("=" * 60)
    print("EXP012: Feature Selection from EXP003")
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
    print(f"\nTotal features extracted: {len(feature_cols)}")

    y_train = sample_features['abs_focus'].values
    patterns = sample_features['pattern'].values

    # ============================================
    # 特徴量数の探索
    # ============================================
    print("\n[3/4] Searching optimal number of features...")

    results = []
    for n_features in [10, 15, 20, 25, 30, 40, 50, 60]:
        cv_rmse, selected, oof_preds = train_and_evaluate(
            sample_features, y_train, feature_cols, patterns, n_features
        )
        results.append({
            'n_features': len(selected),
            'cv_rmse': cv_rmse,
            'features': selected
        })
        print(f"  n={len(selected):2d}: CV RMSE = {cv_rmse:.4f}")

    # 最適な特徴量数を選択（CV RMSEが最小）
    best_result = min(results, key=lambda x: x['cv_rmse'])
    best_n = best_result['n_features']
    best_features = best_result['features']

    print(f"\nBest: n={best_n} features, CV RMSE={best_result['cv_rmse']:.4f}")
    print(f"\nSelected features:")
    for i, f in enumerate(best_features):
        print(f"  {i+1}. {f}")

    # ============================================
    # 最適な特徴量数で最終モデルを学習
    # ============================================
    print(f"\n[4/4] Training final model with {best_n} features...")

    X_train = sample_features[best_features].values
    X_test = test_features[best_features].values

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

    # CV for final evaluation
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
    print(f"\nFinal CV RMSE: {cv_rmse:.4f}")

    # パターン別RMSE
    print("\nPattern-wise CV RMSE:")
    for pattern in sorted(sample_features['pattern'].unique()):
        mask = patterns == pattern
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
        'feature': best_features,
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    # テストデータ予測
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # CV RMSE vs n_features
    axes[0, 0].plot([r['n_features'] for r in results], [r['cv_rmse'] for r in results], 'bo-')
    axes[0, 0].axvline(x=best_n, color='r', linestyle='--', label=f'Best: n={best_n}')
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('CV RMSE')
    axes[0, 0].set_title('CV RMSE vs Number of Features')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # True vs Predicted
    axes[0, 1].scatter(y_train, oof_preds, alpha=0.7)
    axes[0, 1].plot([0, 100], [0, 100], 'r--')
    axes[0, 1].set_xlabel('True abs_focus')
    axes[0, 1].set_ylabel('Predicted abs_focus')
    axes[0, 1].set_title(f'Sample: True vs OOF Predicted (RMSE={cv_rmse:.2f})')

    # Test prediction distribution
    axes[1, 0].hist(y_pred_test, bins=30, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('abs_focus')
    axes[1, 0].set_title('Test: Predicted Distribution')

    # Feature importance
    axes[1, 1].barh(importance['feature'], importance['importance'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Feature Importance')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_distribution.png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'prediction_distribution.png'}")
    plt.close()

    # 結果保存
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # 探索結果保存
    search_results = pd.DataFrame([
        {'n_features': r['n_features'], 'cv_rmse': r['cv_rmse']}
        for r in results
    ])
    search_results.to_csv(OUTPUT_DIR / "feature_search_results.csv", index=False)


if __name__ == '__main__':
    main()
