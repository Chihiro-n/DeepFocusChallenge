"""
EXP000: ボケ検出用特徴量の抽出テスト

特徴量:
1. Laplacian分散: ボケ→エッジ弱→分散小
2. FFT高周波成分: ボケ→高周波減少
3. 勾配統計量 (Sobel/Scharr): ボケ→勾配弱

使用方法:
```
python EXP/EXP000/test_blur_features.py --image_path "input/DeepFocusChallenge_v2/sample/020000.JPG"

# 複数画像を比較
python EXP/EXP000/test_blur_features.py --image_paths "input/DeepFocusChallenge_v2/sample/020000.JPG" "input/DeepFocusChallenge_v2/sample/020005.JPG" "input/DeepFocusChallenge_v2/sample/020010.JPG"
```
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def extract_blur_features(image_path: str) -> Dict[str, float]:
    """
    画像からボケ検出用の特徴量を抽出
    """
    # グレースケールで読み込み
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_float = img.astype(np.float64)
    features = {}

    # ============================================
    # 1. Laplacian 分散
    # ============================================
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features['laplacian_var'] = float(laplacian.var())
    features['laplacian_std'] = float(laplacian.std())
    features['laplacian_mean_abs'] = float(np.abs(laplacian).mean())

    # ============================================
    # 2. FFT 高周波成分
    # ============================================
    # 2D FFT
    f = np.fft.fft2(img_float)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 中心からの距離でマスク作成
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # 低周波・中周波・高周波の分離
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

    # 周波数帯域の定義（画像サイズの比率で）
    max_dist = np.sqrt(crow ** 2 + ccol ** 2)
    low_thresh = max_dist * 0.1
    mid_thresh = max_dist * 0.3

    low_mask = dist_from_center <= low_thresh
    mid_mask = (dist_from_center > low_thresh) & (dist_from_center <= mid_thresh)
    high_mask = dist_from_center > mid_thresh

    # 各帯域のエネルギー
    total_energy = np.sum(magnitude ** 2)
    low_energy = np.sum((magnitude * low_mask) ** 2)
    mid_energy = np.sum((magnitude * mid_mask) ** 2)
    high_energy = np.sum((magnitude * high_mask) ** 2)

    features['fft_high_energy'] = float(high_energy)
    features['fft_high_ratio'] = float(high_energy / total_energy) if total_energy > 0 else 0
    features['fft_mid_ratio'] = float(mid_energy / total_energy) if total_energy > 0 else 0
    features['fft_low_ratio'] = float(low_energy / total_energy) if total_energy > 0 else 0
    features['fft_high_to_low'] = float(high_energy / low_energy) if low_energy > 0 else 0

    # ============================================
    # 3. 勾配統計量 (Sobel)
    # ============================================
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    features['sobel_mean'] = float(sobel_mag.mean())
    features['sobel_std'] = float(sobel_mag.std())
    features['sobel_max'] = float(sobel_mag.max())
    features['sobel_p95'] = float(np.percentile(sobel_mag, 95))

    # ============================================
    # 4. Scharr (より精度の高い勾配)
    # ============================================
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    features['scharr_mean'] = float(scharr_mag.mean())
    features['scharr_std'] = float(scharr_mag.std())

    # ============================================
    # 5. Canny エッジ密度
    # ============================================
    # 複数の閾値で試す
    edges_low = cv2.Canny(img, 50, 150)
    edges_high = cv2.Canny(img, 100, 200)

    features['canny_density_low'] = float(edges_low.mean() / 255)
    features['canny_density_high'] = float(edges_high.mean() / 255)

    # ============================================
    # 6. 局所コントラスト
    # ============================================
    # 標準偏差フィルタ
    kernel_size = 5
    local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    features['local_contrast_mean'] = float(local_std.mean())
    features['local_contrast_std'] = float(local_std.std())

    return features


def visualize_features(image_path: str, features: Dict[str, float], save_path: str = None):
    """特徴量の可視化"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 元画像
    axes[0, 0].imshow(img_color)
    axes[0, 0].set_title(f"Original\n{Path(image_path).name}")
    axes[0, 0].axis('off')

    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    axes[0, 1].imshow(np.abs(laplacian), cmap='hot')
    axes[0, 1].set_title(f"Laplacian\nvar={features['laplacian_var']:.2f}")
    axes[0, 1].axis('off')

    # Sobel magnitude
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    axes[0, 2].imshow(sobel_mag, cmap='hot')
    axes[0, 2].set_title(f"Sobel Magnitude\nmean={features['sobel_mean']:.2f}")
    axes[0, 2].axis('off')

    # Canny edges
    edges = cv2.Canny(img, 50, 150)
    axes[0, 3].imshow(edges, cmap='gray')
    axes[0, 3].set_title(f"Canny Edges\ndensity={features['canny_density_low']:.4f}")
    axes[0, 3].axis('off')

    # FFT magnitude spectrum
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    axes[1, 0].imshow(magnitude, cmap='gray')
    axes[1, 0].set_title(f"FFT Magnitude\nhigh_ratio={features['fft_high_ratio']:.4f}")
    axes[1, 0].axis('off')

    # Local contrast
    kernel_size = 5
    local_mean = cv2.blur(img.astype(np.float64), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img.astype(np.float64) ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    axes[1, 1].imshow(local_std, cmap='hot')
    axes[1, 1].set_title(f"Local Contrast\nmean={features['local_contrast_mean']:.2f}")
    axes[1, 1].axis('off')

    # 主要特徴量のバーチャート
    key_features = ['laplacian_var', 'sobel_mean', 'fft_high_ratio', 'canny_density_low']
    values = [features[k] for k in key_features]
    axes[1, 2].barh(key_features, values, color='steelblue')
    axes[1, 2].set_title("Key Features")
    axes[1, 2].set_xlim(0, max(values) * 1.2 if max(values) > 0 else 1)

    # 全特徴量テキスト
    axes[1, 3].axis('off')
    text = "All Features:\n" + "\n".join([f"{k}: {v:.4f}" for k, v in sorted(features.items())])
    axes[1, 3].text(0.1, 0.95, text, transform=axes[1, 3].transAxes,
                    fontsize=8, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def compare_images(image_paths: List[str]):
    """複数画像の特徴量を比較"""
    all_features = []
    names = []

    for path in image_paths:
        features = extract_blur_features(path)
        features['image'] = Path(path).name
        all_features.append(features)
        names.append(Path(path).name)

    # 比較表示
    print("\n" + "=" * 80)
    print("Feature Comparison")
    print("=" * 80)

    # 主要特徴量のみ表示
    key_features = ['laplacian_var', 'sobel_mean', 'sobel_std', 'fft_high_ratio',
                    'canny_density_low', 'local_contrast_mean']

    # ヘッダー
    header = f"{'Feature':<25}" + "".join([f"{name:<15}" for name in names])
    print(header)
    print("-" * 80)

    for feat in key_features:
        row = f"{feat:<25}"
        for f in all_features:
            row += f"{f[feat]:<15.4f}"
        print(row)

    # グラフで比較
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        values = [f[feat] for f in all_features]
        bars = axes[idx].bar(names, values, color='steelblue')
        axes[idx].set_title(feat)
        axes[idx].tick_params(axis='x', rotation=45)

        # 値をバーの上に表示
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    return all_features


def main():
    parser = argparse.ArgumentParser(description="Extract blur detection features from images")
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_paths', type=str, nargs='+', help='Paths to multiple images for comparison')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()

    if args.image_paths:
        # 複数画像の比較
        compare_images(args.image_paths)
    elif args.image_path:
        # 単一画像の解析
        print(f"Processing: {args.image_path}")
        features = extract_blur_features(args.image_path)

        print("\nExtracted Features:")
        print("-" * 40)
        for key, value in sorted(features.items()):
            print(f"  {key:<25}: {value:.6f}")

        visualize_features(args.image_path, features, args.save_path)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
