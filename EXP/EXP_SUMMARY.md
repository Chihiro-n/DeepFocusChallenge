# Experiment Summary

## 全体サマリー

| EXP | アプローチ | CV RMSE | LB Score |
|-----|-----------|---------|----------|
| EXP000 | Ridge (3特徴量) | 20.46 | 30.05 |
| EXP001 | LightGBM (33特徴量) | 11.55 | 23.65 |
| EXP002 | Train基準Z-score + Ridge | 17.25 | 19.73 |
| **EXP003** | **LightGBM + DeepResearch (60特徴量)** | **11.35** | **19.16** ⭐ |
| EXP004 | EXP002+EXP003融合 (120特徴量) | 11.96 | 19.74 |
| EXP005 | Monotone Constraints LightGBM | 11.72 | 19.84 |
| EXP007 | Reblur-ratio Features (104特徴量) | 11.36 | 19.75 |
| EXP009 | Patch-wise FM Distribution (230特徴量) | 9.69 | 20.95 |
| EXP010 | Contrast-Invariant Features (81特徴量) | 11.31 | 20.32 |
| EXP011 | FOV-Normalized Features (99特徴量) | 11.76 | 20.27 |
| EXP012 | Feature Selection (15特徴量) | 10.96 | 20.14 |
| EXP013 | EXP003 + Blur Add-ons (65特徴量) | 11.28 | 19.24 |
| EXP014 | Ridge + GroupKFold (Discussion版) | TBD | TBD |

---

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
| child-exp000 | 33 | 11.55 | 23.65 | EXP000から6.4pt改善 |

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
| child-exp000 | Train全体 | Ridge | 17.25 | 19.73 | EXP001から3.9pt改善 |

### 結果・知見

1. **Train基準のZ-score化が有効** - Sample RMSE 17.25 → LB 19.73（差が小さい＝汎化性能が高い）
2. **Feature coefficients**:
   - `scharr_std`: +20.2（正の寄与、想定外）
   - `sobel_max_zscore`: -12.4（Z-score化が効いている）
   - `local_contrast_std_k7`: -5.6
3. **Pattern別RMSE**:
   - Pattern 4: 9.97（最良）
   - Pattern 1: 23.83（依然として最悪）
4. **条件の違い**:
   - Train: `2_1.5_10Pa`, `2_1.5_5Pa` のみ
   - Test: より多様な条件（`2_2.0_10Pa`, `2_2.0_High`等）
   - Train基準でも未知条件に対応できている

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
| child-exp000 | 60 | 11.35 | **19.16** | **1st place**, DeepResearch特徴量が効果的 |

### 結果・知見

1. **DeepResearch特徴量が上位を占める**:
   - `mtf_decay_high_to_vhigh` (101) - MTF減衰率
   - `jnb_sharp_ratio_t30` (100) - JNBシャープネス
   - `log_var_s5.0` (78) - Multi-scale LoG
2. **Pattern別RMSE**:
   - Pattern 4: 5.32（最良、大幅改善）
   - Pattern 1: 15.36（依然最悪だが改善）
3. **EXP001との比較**:
   - 同じLightGBMでCV RMSEは同程度(11.35 vs 11.55)
   - LBは大幅改善(19.16 vs 23.65) → DeepResearch特徴量の汎化性能が高い

---

## EXP004: Train基準Z-score + DeepResearch + LightGBM

### 仮説

EXP002とEXP003の融合により、さらなる精度向上を狙う:
- EXP002: Train基準Z-score化 → 条件変動にロバスト
- EXP003: DeepResearch特徴量 → 物理的に意味のある特徴量

### 手法

1. Train画像(2220枚)の特徴量分布を計算
2. DeepResearch特徴量（60個）を抽出
3. 各特徴量をTrain基準でZ-score化
4. 元特徴量 + Z-score特徴量（120個）でLightGBM

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 120 | 11.96 | 19.74 | EXP003より悪化 |

### 結果・知見

1. **融合は逆効果** - EXP003 (19.16) より悪化 (19.74)
2. **CV RMSEも悪化** - 11.35 → 11.96
3. **考察**:
   - Train条件が限定的（`2_1.5_10Pa`, `2_1.5_5Pa`のみ）
   - Test/Sampleの条件と異なるため、Z-score化がノイズになった可能性
   - EXP002では効果的だったが、DeepResearch特徴量との組み合わせでは機能せず
4. **Feature Importance**: EXP003と同様にJNB系が上位だが、Z-score特徴量は上位に入らず

---

## EXP005: Monotone Constraints LightGBM

### 仮説

EXP003をベースに、物理的に正しい単調制約を追加。
「ボケが強いほど単調に特徴量が変化する」という物理法則を制約として入れる。

### 手法

1. EXP003と同じDeepResearch特徴量（60個）を使用
2. 各特徴量に単調制約を設定:
   - **-1 (シャープ指標)**: `laplacian_var`, `sobel_*`, `tenengrad_*`, `local_contrast_*`, `jnb_*` など
   - **+1 (ボケ指標)**: `fft_low_ratio`, `mtf_decay_*`, `esf_width_*` など
   - **0 (制約なし)**: `img_mean`, `img_std`, `grad_entropy` など
3. LightGBMの `monotone_constraints` パラメータで制約を適用

### 期待効果

- 物理的に不整合な予測を抑制
- オーバーフィット防止による汎化性能向上

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 60 | 11.72 | 19.84 | 単調制約が強すぎて悪化 |

### 結果・知見

1. **単調制約は逆効果** - EXP003 (19.16) より悪化 (19.84)
2. **制約が強すぎた**: 45個の特徴量に-1制約 → 表現力低下
3. **Pattern別RMSE**:
   - Pattern 1: 13.75（EXP003の15.36から改善）
   - Pattern 4: 5.88（EXP003の5.32から悪化）
4. **Feature Importance**: 制約なし(0)の`img_std`がトップ（190）
   - 制約ありの特徴量より制約なしが使われている
5. **考察**: 物理的制約は正しい方向だが、全特徴量に厳密に適用するのは過剰

---

## EXP007: Reblur-ratio Features

### 仮説

追加ぼかしを適用して、元画像との特徴量差分を計算する。
既にボケている画像は追加ぼかしを加えても変化が少ない。

原理:
- **シャープな画像**: ぼかすと特徴量が大きく変化
- **ボケた画像**: 追加ぼかしでも変化が少ない（既に低周波成分のみ）

### 手法

1. 元画像から特徴量抽出 (F_orig)
2. Gaussianぼかしを適用（σ = 2.0, 3.0, 5.0）
3. ぼかし画像から特徴量抽出 (F_blur)
4. 差分・比率を計算:
   - `diff = F_orig - F_blur` (シャープな画像ほど大きい)
   - `ratio = F_blur / F_orig` (ボケた画像ほど1に近い)
5. DeepResearch特徴量 + Reblur特徴量でLightGBM

### 期待効果

- ドメイン不変: 装置条件に依存しない相対的な指標
- Pattern1問題への対処: 輝度ベースではなく変化量ベース

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 104 | 11.36 | 19.75 | 特徴量増加でオーバーフィット |

### 結果・知見

1. **CV同等だがLB悪化** - 特徴量増加（60→104）によるオーバーフィット
2. **Reblur特徴量の重要度は低い**: Top 15に2つのみ
   - `reblur_ratio_sobel_max_s2.0` (52)
   - `reblur_ratio_laplacian_var_s2.0` (42)
3. **既存特徴量が依然として支配的**: `jnb_sharp_ratio_t30` (108)がトップ
4. **Pattern別RMSE**:
   - Pattern 1: 14.69（EXP003の15.36から改善）
   - Pattern 4: 5.27（EXP003と同等）
5. **考察**: Reblurアプローチは補助的な情報を提供するが、主役にはなれない

---

## EXP009: Patch-wise Focus Measure Distribution

### 仮説

画像をパッチに分割し、パッチごとのフォーカス指標の分布を特徴量化。
全体統計量だけでは局所的な品質差が消されてしまう問題に対処。

Pattern1（明暗パターン）問題:
- 明部と暗部でフォーカス品質が異なる可能性
- 全体平均だと局所的なばらつきが消える
- パッチ単位の分布を見ることで不均一性を検出

### 手法

1. 画像を複数サイズのパッチに分割（32x32, 64x64, 128x128）
2. 各パッチのフォーカス指標を計算:
   - Laplacian variance
   - Sobel mean
   - Tenengrad
   - Local contrast
   - Gradient energy
3. パッチ間の分布統計量を計算:
   - mean, std, min, max, range
   - skew, kurtosis, CV
   - p10, p90, IQR
4. 4分割（象限）のFM値と差分も追加
5. DeepResearch特徴量 + パッチ特徴量でLightGBM

### 期待効果

- 局所的なフォーカス不均一を検出
- Pattern1のような特殊パターンへの対応
- 空間的な情報の活用

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 230 | 9.69 | 20.95 | CV改善だがLB悪化（オーバーフィット） |

### 結果・知見

1. **CV大幅改善だがLB悪化** - 典型的なオーバーフィット
   - CV: 11.35 → 9.69（改善）
   - LB: 19.16 → 20.95（悪化）
2. **Pattern 1は改善**: 15.36 → 12.51
3. **Feature Importance**:
   - `quad_ud_diff` (121) が圧倒的トップ → Sample特有の特徴を学習
   - パッチ特徴量が上位を占める
4. **考察**:
   - 230特徴量は多すぎる
   - パッチ分割アプローチは方向性は正しいが、汎化性能が低い
   - Sampleの55枚に対して特徴量が過剰

---

## EXP010: Contrast-Invariant Features

### 仮説

Gemini DeepResearchの調査結果に基づき、「コントラスト不変」の特徴量を追加。
Pattern 1問題の根本原因が「コントラスト/輝度依存性」である可能性に対処。

### 追加特徴量

| カテゴリ | 特徴量 | 特性 |
|---------|--------|------|
| Spectral Kurtosis | `spectral_kurtosis`, `spectral_skewness`, etc. | コントラスト不変（無次元量） |
| Autocorrelation | `autocorr_h`, `autocorr_v`, `autocorr_total_norm` | ノイズ耐性、Vollahの方法 |
| LBP | `lbp_entropy`, `lbp_uniformity`, `lbp_non_uniform_ratio` | 照明不変（大小関係のみ依存） |
| Normalized Variance | `normalized_variance`, `cv` | 輝度変動補正 |

### 期待効果

- **コントラスト不変性**: 画像の明るさ・コントラストが変わっても安定
- **Pattern 1対策**: 輝度依存の特徴量が効かないPattern 1を攻略
- **ノイズ耐性**: Autocorrelationによる安定した測定

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 81 | 11.31 | 20.32 | LB悪化、新特徴量は補助的 |

### 結果・知見

1. **CV同等だがLB悪化** - 特徴量増加による影響
2. **新規特徴量の重要度**: `lbp_mean` (50), `autocorr_v` (44) がTop 15入り
3. **既存特徴量が依然支配的**: `jnb_sharp_ratio_t30` (100) がトップ
4. **Pattern 1**: 13.88（EXP003: 15.36から改善）

---

## EXP011: FOV-Normalized Features

### 仮説

FOV（Field of View）によるスケール差がドメインシフトの原因である可能性。

**問題:**
- Train: FOV=2 のみ
- Sample: FOV=2, 4（Pattern 2はFOV=4）
- Test: FOV=2, 3, 4（多様）

**影響:**
- FOV 2um: 1ピクセル = 2/1024 um
- FOV 4um: 1ピクセル = 4/1024 um（2倍粗い）
- 同じ物理的エッジでもFOV 4の画像はSobel値が半分になる

### 手法

1. `pixel_size = FOV / ImageSize` を計算
2. `scale_factor = pixel_size / base_pixel_size`（FOV=2を基準）
3. 勾配系特徴量を `scale_factor` で正規化:
   - 一階微分（Sobel等）: `feature * scale_factor`
   - 二階微分（Laplacian等）: `feature * scale_factor^2`
4. 生の特徴量 + 正規化特徴量の両方を使用

### 期待効果

- FOVに依存しない物理単位での特徴量
- Train/Test間のドメインギャップを解消
- 特にPattern 2（FOV=4）の精度向上

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 99 | 11.76 | 20.27 | LB悪化、正規化特徴量よりも生が優先される |

### 結果・知見

1. **CV同等だがLB悪化** - EXP003 (19.16) → 20.27
2. **Feature Importance**: 正規化特徴量よりも生の特徴量が上位
   - モデルは自動的に生の特徴量を優先
   - FOV正規化は意図した効果を発揮せず
3. **Pattern別RMSE**:
   - Pattern 1: 13.99（EXP003: 15.36から改善）
   - Pattern 2: 9.46（FOV=4だが改善せず）
4. **考察**:
   - 特徴量増加によるオーバーフィットが支配的
   - FOV正規化はTrain/Testのドメインシフト解消に不十分

---

## EXP012: Feature Selection from EXP003

### 仮説

EXP003の60特徴量から重要度の高いものだけを選択し、オーバーフィットを軽減。
55サンプルに対して60特徴量は多すぎる可能性がある。

### 手法

1. EXP003と同じ60特徴量を抽出
2. EXP003のFeature Importanceに基づき上位N個を選択
3. 複数のN（10, 15, 20, 25, 30, 40, 50, 60）でCV RMSEを比較
4. 最適なN（=15）を選択

### 選択された15特徴量

1. `mtf_decay_high_to_vhigh` - MTF減衰率
2. `jnb_sharp_ratio_t30` - JNBシャープネス比率
3. `log_var_s5.0` - LoG分散（σ=5）
4. `img_std` - 画像標準偏差
5. `fft_low_ratio` - 低周波エネルギー比率
6. `jnb_sharp_ratio_t10` - JNBシャープネス比率
7. `log_max_s5.0` - LoG最大値（σ=5）
8. `grad_skewness` - 勾配歪度
9. `fft_mid_ratio` - 中周波エネルギー比率
10. `edge_gradient_p25` - エッジ勾配25パーセンタイル
11. `jnb_sharp_ratio_t20` - JNBシャープネス比率
12. `laplacian_var` - ラプラシアン分散
13. `log_max_s3.0` - LoG最大値（σ=3）
14. `mtf_low_high_ratio` - 低周波/高周波比
15. `edge_gradient_mean` - エッジ勾配平均

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 15 | 10.96 | 20.14 | CV改善だがLB改善せず |

### 結果・知見

1. **CV改善だがLB悪化** - 典型的な方向違い
   - CV: 11.35 → 10.96（改善）
   - LB: 19.16 → 20.14（悪化）
2. **特徴量数別CV RMSE**:
   - n=10: 11.96
   - n=15: **10.96**（最良）
   - n=20: 11.93
   - n=25: 11.16
   - n=30: 11.50
   - n=60: 11.82
3. **Pattern別RMSE**:
   - Pattern 0: 12.85
   - Pattern 1: 13.10
   - Pattern 2: 8.77
   - Pattern 3: 12.72
   - Pattern 4: 4.99（最良）
4. **考察**:
   - 特徴量を減らしてもLBは改善しなかった
   - CV RMSEとLBの相関が低い（CV改善≠LB改善）
   - EXP003の60特徴量がバランス良く機能している可能性
   - 問題はオーバーフィットではなく、ドメインシフトの可能性

---

## EXP013: EXP003 + Blur Add-ons

### 仮説

EXP003にボケ特化の追加指標を重ねることで、ドメイン差に左右されない高周波減衰の捕捉力を高める。
- パワースペクトル重心/半径（低周波寄り度合い）
- LoGピラミッド減衰傾き（スケール間の高周波落ち方）
- 弱リブラー差分（σ=1.5での追加ぼかしによる相対減少）

### 手法

1. EXP003の全60特徴量を継承
2. 追加特徴（計5本）を導入: `power_centroid_norm`, `power_radius_p50_norm`, `log_pyramid_slope`, `reblur_lap_var_ratio_s1p5`, `reblur_sobel_p95_drop_s1p5`
3. LightGBMはEXP003と同設定（num_leaves=15, max_depth=4, 500r/ES）

### 実験

| Child Exp | 特徴量数 | CV RMSE | LB Score | 備考 |
|-----------|----------|---------|----------|------|
| child-exp000 | 65 | 11.28 | 19.24 | EXP003よりわずかに悪化 |

### 結果・知見

1. **LBは僅差で悪化** - EXP003 (19.16) → 19.24
2. **CVはわずかに改善** - 11.35 → 11.28 だがLBに未反映
3. **Pattern別RMSE**:
   - Pattern 1: 14.80（EXP003: 15.36から改善）
   - Pattern 4: 5.07（EXP003: 5.32から改善）
4. **Feature Importance**: 追加特徴はTop15に入らず（MTF/JNB/LoGが引き続き支配）
5. **考察**:
   - 追加指標は方向性は悪くないが、寄与が小さい
   - Pattern1/4は改善傾向だが総合LBはほぼ横ばい
   - さらなる効果には特徴量のスケーリング/組み合わせ調整が必要

---

## EXP014: Ridge + GroupKFold (Discussion改善版)

### 仮説

Discussionで共有されたLB 22.3のシンプルなコードから得られた知見:
1. **過学習問題**: CatBoostでGroupKFoldするとCV 20.21、LB 24（乖離が大きい）
2. **データ量不足**: 55枚のSampleに対してLightGBMは複雑すぎる可能性
3. **シンプルなモデル**: Ridge回帰など正則化の強いモデルが有効

### 手法

1. **GroupKFold**: patternをグループとしてCV分割（パターン間汎化を正確に評価）
2. **シンプルなモデル**: Ridge, Lasso, ElasticNetを比較
3. **Discussion版特徴量**: 周波数帯域のmean + std（5帯域×2）
4. **EXP003から厳選**: Multi-scale LoG, Tenengrad, Power spectrum slopeなど

### 特徴量（22個）

| カテゴリ | 特徴量 |
|---------|--------|
| Laplacian | `laplacian_var`, `laplacian_mean_abs` |
| FFT (5帯域×2) | `fft_band_{r}_mean`, `fft_band_{r}_std` |
| Frequency Ratio | `high_low_freq_ratio` |
| Sobel | `sobel_mean`, `sobel_std`, `sobel_p95` |
| Local Contrast | `local_contrast_mean` (kernel=32) |
| Multi-scale LoG | `log_var_s{1,2,3}` |
| Tenengrad | `tenengrad_mean` |
| Power Spectrum | `spectrum_slope` |
| Image Stats | `img_mean`, `img_std` |

### 実験

| Child Exp | モデル | 特徴量数 | GroupKFold Macro RMSE | LB Score | 備考 |
|-----------|--------|----------|----------------------|----------|------|
| child-exp000 | Ridge (α=10) | 22 | TBD | TBD | Discussion改善版 |

### 結果・知見

（実行後に記入）
