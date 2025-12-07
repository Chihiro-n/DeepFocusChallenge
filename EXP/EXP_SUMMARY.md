# Experiment Summary

## EXP000: ボケ特徴量による defocus 推定

### 仮説（初期）

Depth Anything v3 で depth map を生成し、凹凸の大きさから推定する。
→ **棄却**: SEM画像では期待通りに動作せず、逆の相関を示した。

### 方針転換

画像処理ベースのボケ検出特徴量を使用:
- Laplacian分散
- FFT周波数成分
- Sobel/Scharr勾配
- 局所コントラスト

### 重要な発見

1. **全体相関とパターン内相関は異なる**
   - `img_mean`: 全体相関0.79だがPattern1では相関なし → 使えない
   - `local_contrast_std`: 全Pattern一貫して負の相関 → 信頼できる

2. **有効な特徴量**（パターン内で一貫した負の相関）:
   - `local_contrast_std` (r = -0.94〜-0.99)
   - `fft_mid_ratio` (r = -0.88〜-0.99)
   - `sobel_max` (r = -0.78〜-0.96)

### 実験

| Child Exp | 特徴量 | モデル | Sample RMSE | LB Score | 備考 |
|-----------|--------|--------|-------------|----------|------|
| child-exp000 | local_contrast_std, fft_mid_ratio, sobel_max | Ridge | 20.46 | **30.05** | **1st place** |

---

## EXP001: LightGBM による非線形モデル

### 仮説

Ridge回帰（線形）では捉えられない非線形関係をLightGBMで学習する。
特徴量も拡張（複数カーネルサイズ、複数周波数帯域など）。

### 手法

1. 拡張特徴量抽出（40+特徴量）
2. LightGBMで5-fold CV
3. 全データで再学習してTest予測

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 40+ | - | - | 初回実験 |

### 結果・知見

（実験実行後に記載）

---

## EXP002: Train画像を基準としたdefocus推定

### 仮説

Train画像（2220枚）は主にabs_focus=0（ベストフォーカス）。
これを「フォーカスが合っている状態」の基準として使用し、
Test画像の特徴量が基準からどれだけ離れているかでdefocusを推定する。

### 手法

1. Train画像の特徴量分布を計算（mean, std）
2. Sample/Test画像の特徴量を基準との差分（Z-score, diff）に変換
3. Sample画像で差分特徴量とabs_focusの関係を学習
4. Test画像に適用

### 実験

| Child Exp | 基準 | モデル | Sample RMSE | LB Score | 備考 |
|-----------|------|--------|-------------|----------|------|
| child-exp000 | Train全体 | Ridge | - | - | 初回実験 |

### 結果・知見

（実験実行後に記載）

---
