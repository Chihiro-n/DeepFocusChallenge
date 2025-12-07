import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# 1. ExG + Otsu で vegetation mask を作る
# ---------------------------------------------------------
def compute_veg_mask_exg(img_bgr: np.ndarray):
    """
    入力:
        img_bgr: OpenCV形式の BGR 画像 (H, W, 3), uint8
    出力:
        exg_norm: 0〜1 に正規化した Excess Green マップ (float32)
        veg_mask: vegetation / non-vegetation の2値マスク (uint8, 0 or 1)
    """
    img = img_bgr.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)

    # Excess Green
    exg = 2 * g - r - b

    # 0〜1へ正規化
    exg_min, exg_max = exg.min(), exg.max()
    exg_norm = (exg - exg_min) / (exg_max - exg_min + 1e-6)

    # 8bitにして大津の二値化で vegetation / non-vegetation を分離
    exg_8u = (exg_norm * 255).astype(np.uint8)
    _, th = cv2.threshold(exg_8u, 0, 255, cv2.THRESH_OTSU)

    veg_mask = (exg_8u >= th).astype(np.uint8)  # 1: vegetation, 0: その他
    return exg_norm.astype(np.float32), veg_mask


# ---------------------------------------------------------
# 2. HSV 情報を使って live / dead / soil をざっくり分類
# ---------------------------------------------------------
def classify_live_dead_soil(img_bgr: np.ndarray, veg_mask: np.ndarray):
    """
    入力:
        img_bgr: BGR 画像, uint8
        veg_mask: 1=vegetation, 0=その他 のマスク
    出力:
        class_map: 0=soil/other, 1=live green, 2=dead/brown
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = h / 179.0  # 0〜1 に正規化 (OpenCVのHは0〜179度)
    s = s / 255.0
    v = v / 255.0

    # 初期は soil/other = 0
    class_map = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    # vegetation 領域だけを対象
    veg = veg_mask.astype(bool)

    # 緑っぽい（live）: Hが緑〜黄緑 / 彩度・明度そこそこ
    live_cond = (
        (h >= 0.20) & (h <= 0.45) &  # ざっくり緑〜黄緑
        (s >= 0.25) &
        (v >= 0.30)
    )

    # 枯れ草っぽい（dead）: Hが黄〜茶色寄り or 彩度が低めのくすんだ緑
    dead_cond = (
        (h >= 0.08) & (h <= 0.25) & (v >= 0.20)
    ) | (
        (live_cond == False) & (veg) & (v >= 0.15)
    )

    # veg領域のうち live_cond を満たす → 1
    class_map[veg & live_cond] = 1
    # veg領域のうち dead_cond を満たす → 2
    class_map[veg & dead_cond] = 2
    # veg==0 は soil/other のまま 0

    return class_map


# ---------------------------------------------------------
# 3. 統計値（被覆率など）を計算
# ---------------------------------------------------------
def compute_cover_ratios(veg_mask: np.ndarray, class_map: np.ndarray):
    """
    入力:
        veg_mask: 1=vegetation, 0=その他
        class_map: 0=soil/other, 1=live green, 2=dead/brown
    出力:
        dict で ratio を返す
    """
    h, w = veg_mask.shape
    total = h * w

    veg_ratio = veg_mask.sum() / total

    live_ratio = (class_map == 1).sum() / total
    dead_ratio = (class_map == 2).sum() / total
    soil_ratio = (class_map == 0).sum() / total

    return {
        "veg_ratio": veg_ratio,
        "live_ratio": live_ratio,
        "dead_ratio": dead_ratio,
        "soil_ratio": soil_ratio,
    }


# ---------------------------------------------------------
# 4. 可視化
# ---------------------------------------------------------
def visualize_exg_mask_and_classes(image_path: str, save_prefix: str | None = None):
    """
    任意の画像パスを渡すと、
    - 元画像
    - ExGヒートマップ
    - vegetation mask
    - live/dead/soil クラスマップ
    を描画し、cover ratio をprint。
    save_prefix を指定するとPNGで保存もする。
    """
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1) ExG + veg_mask
    exg_norm, veg_mask = compute_veg_mask_exg(img_bgr)

    # 2) live / dead / soil 分類
    class_map = classify_live_dead_soil(img_bgr, veg_mask)

    # 3) ratio 計算
    ratios = compute_cover_ratios(veg_mask, class_map)
    print("Coverage ratios:")
    for k, v in ratios.items():
        print(f"  {k}: {v:.3f}")

    # 4) 可視化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (1) 元画像
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis("off")

    # (2) ExGヒートマップ
    im1 = axes[0, 1].imshow(exg_norm, cmap="viridis")
    axes[0, 1].set_title("Excess Green (normalized)")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (3) vegetation mask
    axes[1, 0].imshow(veg_mask, cmap="gray")
    axes[1, 0].set_title("Vegetation Mask (ExG + Otsu)")
    axes[1, 0].axis("off")

    # (4) live/dead/soil マップ
    # 0=soil(グレー), 1=live(緑), 2=dead(茶色) に色付け
    h, w = class_map.shape
    color_map_vis = np.zeros((h, w, 3), dtype=np.uint8)

    soil_color = np.array([160, 160, 160], dtype=np.uint8)  # gray
    live_color = np.array([0, 200, 0], dtype=np.uint8)      # green
    dead_color = np.array([180, 120, 50], dtype=np.uint8)   # brown

    color_map_vis[class_map == 0] = soil_color
    color_map_vis[class_map == 1] = live_color
    color_map_vis[class_map == 2] = dead_color

    axes[1, 1].imshow(color_map_vis)
    axes[1, 1].set_title("Live / Dead / Soil (heuristic)")
    axes[1, 1].axis("off")

    fig.suptitle(f"Vegetation Analysis: {image_path.name}", fontsize=14)
    plt.tight_layout()

    if save_prefix is not None:
        out_path = Path(save_prefix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()


# ---------------------------------------------------------
# 5. 使い方例
# ---------------------------------------------------------
if __name__ == "__main__":
    # ここを任意の画像パスに変えてください
    test_image_path = "../../input/csiro-biomass/train/ID482555369.jpg"
    visualize_exg_mask_and_classes(test_image_path, save_prefix="exg_veg_vis.png")
