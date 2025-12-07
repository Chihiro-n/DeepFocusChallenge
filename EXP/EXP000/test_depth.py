"""
EXP000: 動作確認用 - 単一画像の depth map 出力

使用方法 (Google Colab):
```python
!git clone https://github.com/ByteDance-Seed/Depth-Anything-3
%cd Depth-Anything-3
!pip install -e .
%cd ..

# 実行（画像パスを指定）
%run EXP/EXP000/test_depth.py --image_path "input/DeepFocusChallenge_v2/sample/020000.JPG"
```
"""

import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_depth_model(model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
    """Depth Anything v3 モデルをロード"""
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def inference_single(model, image_path: str, input_size: int = 518) -> np.ndarray:
    """単一画像の推論（TTA なし、シンプル版）"""
    with torch.no_grad():
        # image はリストで渡す必要がある
        prediction = model.inference(
            image=[image_path],
            process_res=input_size,
            process_res_method="upper_bound_resize",
            export_dir=None,
            export_format="glb"
        )

    depth = prediction.depth
    if hasattr(depth, 'cpu'):
        depth = depth.cpu().numpy()

    # バッチ次元を除去
    if depth.ndim == 3:
        depth = depth[0]

    return depth


def visualize_depth(image_path: str, depth_map: np.ndarray, save_path: str = None):
    """元画像と depth map を並べて可視化"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 元画像
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original\n{Path(image_path).name}")
    axes[0].axis('off')

    # Depth map (カラー)
    im = axes[1].imshow(depth_map, cmap='inferno')
    axes[1].set_title(f"Depth Map\nmean={depth_map.mean():.4f}, std={depth_map.std():.4f}")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Depth map (Spectral)
    im2 = axes[2].imshow(depth_map, cmap='Spectral')
    axes[2].set_title(f"Depth Map (Spectral)\nrange={depth_map.max()-depth_map.min():.4f}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test Depth Anything v3 on a single image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_name', type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--save_path', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()

    # モデルロード
    print("Loading model...")
    model = load_depth_model(args.model_name)

    # 推論
    print(f"Processing: {args.image_path}")
    depth_map = inference_single(model, args.image_path, args.input_size)

    # 統計量表示
    print(f"\nDepth map shape: {depth_map.shape}")
    print(f"  mean:  {depth_map.mean():.6f}")
    print(f"  std:   {depth_map.std():.6f}")
    print(f"  min:   {depth_map.min():.6f}")
    print(f"  max:   {depth_map.max():.6f}")
    print(f"  range: {depth_map.max() - depth_map.min():.6f}")

    # 可視化
    visualize_depth(args.image_path, depth_map, args.save_path)


if __name__ == '__main__':
    main()
