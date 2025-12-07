"""
EXP000: Depth Anything v3 による defocus 推定

仮説:
SEM画像の半導体凹凸において、ボケているほど凹凸が小さく見える（depth mapのstdが小さくなる）。
Depth Anything v3 で depth map を生成し、その統計量（std, range等）から abs_focus を推定する。

使用方法 (Google Colab):
1. Depth-Anything-3 をクローン & インストール
2. このスクリプトを実行

```python
!git clone https://github.com/ByteDance-Seed/Depth-Anything-3
%cd Depth-Anything-3
!pip install -e .
%cd ..

# 実行
%run EXP/EXP000/infer.py --config EXP/EXP000/config/child-exp000.yaml
```
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ============================================
# Depth Anything v3 推論
# ============================================

def load_depth_model(model_name: str, device: torch.device):
    """Depth Anything v3 モデルをロード"""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {model_name} on {device}")
    return model


def inference_single_image(model, image_path: str, input_size: int = 518) -> np.ndarray:
    """
    単一画像に対して4方向TTAで推論し、平均depth mapを返す
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # 4方向回転
    imgs_horizontal = [img_rgb, np.rot90(img_rgb, k=2)]
    imgs_vertical = [np.rot90(img_rgb, k=1), np.rot90(img_rgb, k=3)]

    temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else '/tmp'
    pid = os.getpid()

    def run_inference(img_list, suffix):
        temp_paths = []
        try:
            for i, img in enumerate(img_list):
                path = os.path.join(temp_dir, f'temp_{pid}_{suffix}_{i}.jpg')
                cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                temp_paths.append(path)

            with torch.no_grad():
                prediction = model.inference(
                    image=temp_paths,
                    process_res=input_size,
                    process_res_method="upper_bound_resize",
                    export_dir=None,
                    export_format="glb"
                )

            d = prediction.depth
            if hasattr(d, 'cpu'):
                d = d.cpu().numpy()
            return d
        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    os.remove(p)

    depths_h = run_inference(imgs_horizontal, 'h')
    depths_v = run_inference(imgs_vertical, 'v')

    processed_depths = []

    # 0度
    d0 = depths_h[0]
    if d0.shape[:2] != (original_h, original_w):
        d0 = cv2.resize(d0, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    processed_depths.append(d0)

    # 180度
    d180 = np.rot90(depths_h[1], k=2)
    if d180.shape[:2] != (original_h, original_w):
        d180 = cv2.resize(d180, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    processed_depths.append(d180)

    # 90度
    d90 = np.rot90(depths_v[0], k=3)
    if d90.shape[:2] != (original_h, original_w):
        d90 = cv2.resize(d90, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    processed_depths.append(d90)

    # 270度
    d270 = np.rot90(depths_v[1], k=1)
    if d270.shape[:2] != (original_h, original_w):
        d270 = cv2.resize(d270, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    processed_depths.append(d270)

    return np.mean(processed_depths, axis=0)


# ============================================
# 特徴量抽出
# ============================================

def extract_depth_features(depth_map: np.ndarray) -> dict:
    """
    Depth map から統計特徴量を抽出

    仮説: ボケている画像ほど凹凸が小さく見えるため、
    depth_std, depth_range が小さくなる
    """
    features = {
        'depth_mean': float(np.mean(depth_map)),
        'depth_std': float(np.std(depth_map)),
        'depth_min': float(np.min(depth_map)),
        'depth_max': float(np.max(depth_map)),
        'depth_range': float(np.max(depth_map) - np.min(depth_map)),
        'depth_p5': float(np.percentile(depth_map, 5)),
        'depth_p95': float(np.percentile(depth_map, 95)),
        'depth_iqr': float(np.percentile(depth_map, 75) - np.percentile(depth_map, 25)),
    }
    return features


def process_images(model, df: pd.DataFrame, image_base_dir: str, input_size: int) -> pd.DataFrame:
    """
    DataFrameの全画像に対してdepth推論と特徴量抽出を行う
    """
    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        # filepath は ./sample/xxx.JPG のような形式
        rel_path = row['filepath'].lstrip('./')
        image_path = os.path.join(image_base_dir, rel_path)

        try:
            depth_map = inference_single_image(model, image_path, input_size)
            if depth_map is not None:
                features = extract_depth_features(depth_map)
                features['id'] = row['id']
                all_features.append(features)
            else:
                print(f"Warning: Could not read {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return pd.DataFrame(all_features)


# ============================================
# 回帰モデル
# ============================================

def fit_regression(sample_features: pd.DataFrame, sample_df: pd.DataFrame, feature_cols: list, alpha: float = 1.0):
    """
    Sample データを使って Ridge 回帰を学習
    """
    merged = sample_features.merge(sample_df[['id', 'abs_focus']], on='id')

    X = merged[feature_cols].values
    y = merged['abs_focus'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    # 学習データでの予測結果
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"Sample RMSE: {rmse:.4f}")

    # 係数を表示
    print("\nFeature coefficients:")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"  {feat}: {coef:.4f}")

    return model, scaler


def predict(model, scaler, features_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    テストデータの予測
    """
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    # 負の値をクリップ
    predictions = np.clip(predictions, 0, None)
    return predictions


# ============================================
# メイン
# ============================================

def main(config_path: str):
    # Config読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Config: {config}")

    # パス設定
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # データ読み込み
    sample_df = pd.read_csv(os.path.join(data_dir, 'sample.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    print(f"Sample: {len(sample_df)} images")
    print(f"Test: {len(test_df)} images")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # モデルロード
    model = load_depth_model(config['model_name'], device)

    # 特徴量抽出
    print("\n=== Processing Sample Images ===")
    sample_features = process_images(model, sample_df, data_dir, config['input_size'])
    sample_features.to_csv(os.path.join(output_dir, 'sample_features.csv'), index=False)

    print("\n=== Processing Test Images ===")
    test_features = process_images(model, test_df, data_dir, config['input_size'])
    test_features.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)

    # 回帰モデル学習
    feature_cols = config['feature_cols']
    print(f"\n=== Training Regression (features: {feature_cols}) ===")
    reg_model, scaler = fit_regression(sample_features, sample_df, feature_cols, alpha=config['ridge_alpha'])

    # テストデータ予測
    print("\n=== Predicting Test Data ===")
    predictions = predict(reg_model, scaler, test_features, feature_cols)

    # 提出ファイル作成
    submission = pd.DataFrame({
        'id': test_features['id'],
        'abs_focus': predictions
    })
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    # 予測分布の確認
    print(f"\nPrediction stats:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std:  {predictions.std():.2f}")
    print(f"  Min:  {predictions.min():.2f}")
    print(f"  Max:  {predictions.max():.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    main(args.config)
