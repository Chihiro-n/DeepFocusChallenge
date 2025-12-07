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
| child-exp000 | local_contrast_std, fft_mid_ratio, sobel_max | Ridge | 20.46 | 30.05 | ベースライン |

---

## EXP001: LightGBM による非線形モデル

### 仮説

Ridge回帰（線形）では捉えられない非線形関係をLightGBMで学習する。
特徴量も拡張（複数カーネルサイズ、複数周波数帯域など）。

### 手法

1. 拡張特徴量抽出（33特徴量）
2. LightGBMで5-fold CV
3. 全データで再学習してTest予測

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 33 | 11.55 | **23.65** | **1st place**, EXP000から6.4pt改善 |

### 結果・知見

1. **CV RMSE 11.55** だが Fold間分散が大きい (6.3〜14.6)
2. **Feature Importance Top 5**:
   - `img_std` (175)
   - `fft_low_ratio` (156)
   - `local_contrast_std_k7` (126)
   - `laplacian_var` (122)
   - `local_contrast_std_k5` (115)
3. **Pattern 1 が依然として最悪** (CV RMSE 15.46) → EXP000の発見と一致
4. LightGBMの非線形性が効いている

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

## EXP003: LightGBM + DeepResearch Features

### 仮説

EXP001をベースに、ChatGPT DeepResearchで調査したゼロショットボケ検出手法の知見を追加。
物理的に意味のある特徴量を追加することで精度向上を狙う。

### 追加特徴量（DeepResearchより）

| カテゴリ | 特徴量 | 根拠 |
|---------|--------|------|
| Multi-scale LoG | `log_var_s{1,2,3,5}`, `log_max_s{1,2,3,5}` | マルチスケールでエッジ応答 |
| Tenengrad | `tenengrad_mean`, `tenengrad_sum` | 古典的フォーカス指標 |
| CPBD-like (JNB) | `edge_gradient_*`, `jnb_sharp_ratio_t{10,20,30,50}` | Just Noticeable Blur |
| Edge Spread Function | `esf_width_h_mean`, `esf_width_h_std` | エッジの広がり幅 |
| MTF-like | `mtf_decay_*`, `mtf_low_high_ratio` | 周波数減衰率 |
| Gradient Histogram | `grad_entropy`, `grad_skewness`, `grad_kurtosis` | 勾配分布の形状 |
| Power Spectrum | `spectrum_slope` | 1/fスペクトルの傾き |

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | ~60 | - | - | 初回実験 |

### 結果・知見

（実験実行後に記載）

---
