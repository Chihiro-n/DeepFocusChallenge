"""
EXP000: Sample全体での特徴量とabs_focusの相関分析

使用方法:
```
python EXP/EXP000/analyze_sample_correlation.py
```
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import seaborn as sns


# データパス
DATA_DIR = Path("input/DeepFocusChallenge_v2")
SAMPLE_CSV = DATA_DIR / "sample.csv"


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
    low_thresh = max_dist * 0.1
    mid_thresh = max_dist * 0.3

    low_mask = dist_from_center <= low_thresh
    mid_mask = (dist_from_center > low_thresh) & (dist_from_center <= mid_thresh)
    high_mask = dist_from_center > mid_thresh

    total_energy = np.sum(magnitude ** 2)
    low_energy = np.sum((magnitude * low_mask) ** 2)
    mid_energy = np.sum((magnitude * mid_mask) ** 2)
    high_energy = np.sum((magnitude * high_mask) ** 2)

    features['fft_high_energy'] = float(high_energy)
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
    features['sobel_p95'] = float(np.percentile(sobel_mag, 95))

    # 4. Scharr
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    features['scharr_mean'] = float(scharr_mag.mean())
    features['scharr_std'] = float(scharr_mag.std())

    # 5. Canny
    edges_low = cv2.Canny(img, 50, 150)
    edges_high = cv2.Canny(img, 100, 200)

    features['canny_density_low'] = float(edges_low.mean() / 255)
    features['canny_density_high'] = float(edges_high.mean() / 255)

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


def main():
    # Sample CSV読み込み
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Sample size: {len(df)}")
    print(f"Patterns: {df['pattern'].unique()}")
    print(f"abs_focus range: {df['abs_focus'].min()} - {df['abs_focus'].max()}")

    # 全画像の特徴量抽出
    all_features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        rel_path = row['filepath'].lstrip('./')
        image_path = str(DATA_DIR / rel_path)

        try:
            features = extract_blur_features(image_path)
            features['id'] = row['id']
            features['abs_focus'] = row['abs_focus']
            features['pattern'] = row['pattern']
            all_features.append(features)
        except Exception as e:
            print(f"Error: {image_path}: {e}")

    features_df = pd.DataFrame(all_features)

    # 特徴量カラム
    feature_cols = [c for c in features_df.columns if c not in ['id', 'abs_focus', 'pattern']]

    # ============================================
    # 相関分析
    # ============================================
    print("\n" + "=" * 80)
    print("Correlation with abs_focus")
    print("=" * 80)

    correlations = []
    for col in feature_cols:
        corr, p_value = stats.pearsonr(features_df[col], features_df['abs_focus'])
        correlations.append({
            'feature': col,
            'correlation': corr,
            'p_value': p_value,
            'abs_corr': abs(corr)
        })

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    print(f"\n{'Feature':<25} {'Correlation':>12} {'P-value':>12}")
    print("-" * 50)
    for _, row in corr_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['feature']:<25} {row['correlation']:>12.4f} {row['p_value']:>12.4e} {sig}")

    # ============================================
    # パターン別相関
    # ============================================
    print("\n" + "=" * 80)
    print("Correlation by Pattern (Top 5 features)")
    print("=" * 80)

    top_features = corr_df.head(5)['feature'].tolist()

    pattern_corr = []
    for pattern in sorted(features_df['pattern'].unique()):
        pattern_data = features_df[features_df['pattern'] == pattern]
        row_data = {'pattern': pattern, 'n': len(pattern_data)}
        for feat in top_features:
            corr, _ = stats.pearsonr(pattern_data[feat], pattern_data['abs_focus'])
            row_data[feat] = corr
        pattern_corr.append(row_data)

    pattern_corr_df = pd.DataFrame(pattern_corr)
    print(pattern_corr_df.to_string(index=False))

    # ============================================
    # 可視化
    # ============================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Top 5特徴量 + 全体相関のヒートマップ
    for idx, feat in enumerate(top_features):
        ax = axes[idx]

        # パターン別に色分け
        for pattern in sorted(features_df['pattern'].unique()):
            pattern_data = features_df[features_df['pattern'] == pattern]
            ax.scatter(pattern_data[feat], pattern_data['abs_focus'],
                      label=f'Pattern {pattern}', alpha=0.7)

        # 回帰直線
        x = features_df[feat]
        y = features_df['abs_focus']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)

        corr = corr_df[corr_df['feature'] == feat]['correlation'].values[0]
        ax.set_xlabel(feat)
        ax.set_ylabel('abs_focus')
        ax.set_title(f'{feat}\nr = {corr:.4f}')
        ax.legend(fontsize=8)

    # 相関係数のバーチャート
    ax = axes[5]
    colors = ['green' if c > 0 else 'red' for c in corr_df['correlation']]
    ax.barh(corr_df['feature'], corr_df['correlation'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation with abs_focus')
    ax.set_title('All Features Correlation')

    plt.tight_layout()
    plt.savefig('EXP/EXP000/sample_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: EXP/EXP000/sample_correlation_analysis.png")
    plt.show()

    # ============================================
    # 特徴量保存
    # ============================================
    features_df.to_csv('EXP/EXP000/sample_features_blur.csv', index=False)
    print(f"Saved: EXP/EXP000/sample_features_blur.csv")

    # ============================================
    # 簡易回帰モデルでRMSE確認
    # ============================================
    print("\n" + "=" * 80)
    print("Simple Linear Regression RMSE (using top feature)")
    print("=" * 80)

    best_feature = corr_df.iloc[0]['feature']
    X = features_df[best_feature].values.reshape(-1, 1)
    y = features_df['abs_focus'].values

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"Best feature: {best_feature}")
    print(f"RMSE (on sample data): {rmse:.4f}")
    print(f"Coefficient: {model.coef_[0]:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    return features_df, corr_df


if __name__ == '__main__':
    features_df, corr_df = main()
