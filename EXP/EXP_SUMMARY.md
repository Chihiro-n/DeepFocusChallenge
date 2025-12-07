# Experiment Summary

## EXP000: Depth Anything v3 による defocus 推定

### 仮説

SEM画像は半導体の凹凸を撮影している。ボケている画像は不鮮明になり、凹凸が小さく見える。

- ボケていない画像 → 凹凸が鮮明 → depth map の分散が大きい
- ボケている画像 → 凹凸が不鮮明 → depth map の分散が小さい

この関係を利用し、Depth Anything v3 で depth map を生成し、その統計量（std, range, iqr）から abs_focus を推定する。

### 手法

1. Depth Anything v3 (DA3NESTED-GIANT-LARGE) で各画像の depth map を生成
2. 4方向TTA（0°, 90°, 180°, 270°）で推論し平均
3. Depth map から特徴量を抽出:
   - `depth_std`: 標準偏差
   - `depth_range`: max - min
   - `depth_iqr`: 四分位範囲
4. Sample データ（55枚、ラベル付き）で Ridge 回帰を学習
5. Test データに適用して予測

### 実験

| Child Exp | 特徴量 | Ridge Alpha | Sample RMSE | LB Score | 備考 |
|-----------|--------|-------------|-------------|----------|------|
| child-exp000 | std, range, iqr | 1.0 | - | - | 初回実験 |

### 結果・知見

（実験実行後に記載）

---
