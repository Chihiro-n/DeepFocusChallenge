SEM画像におけるボケ推定のための特徴抽出手法に関する包括的調査報告書
1. 序論：走査電子顕微鏡における結像と劣化の物理的メカニズム
走査電子顕微鏡（SEM: Scanning Electron Microscope）における画像の鮮鋭度（シャープネス）およびボケ（Blur）の推定は、半導体製造における欠陥検査、材料科学における微細構造解析、そして生物学的なナノ構造観察において極めて重要な課題である。光学顕微鏡とは異なり、SEMにおけるボケの生成要因は多岐にわたり、電子ビームの物理的特性、試料との相互作用、および検出器の特性が複雑に絡み合っている。本報告書では、SEM画像のボケを定量的に推定するための特徴抽出機（Feature Extractor）について、古典的な空間フィルタリングから最新の深層学習モデルまでを網羅的に調査し、その有効性と限界を論じる。

1.1 SEMにおけるボケの特異性
SEM画像の画質劣化は、単なる焦点外れ（Defocus）のみならず、非点収差（Astigmatism）、帯電（Charging）、および信号対雑音比（SNR）の低下によって引き起こされる。

ビーム径と相互作用体積: SEMの空間分解能は、試料表面上の電子ビームスポット径と、試料内部での電子散乱による相互作用体積（Interaction Volume）によって決定される。焦点が合致していない場合、ビーム径が拡大し、隣接するピクセル間での信号の畳み込み（Convolution）が発生する。これは光学的なガウシアンボケに類似しているが、電子レンズのヒステリシスやビーム電流の変動により、より確率的な挙動を示す 。   

非点収差による指向性ボケ: 電磁レンズの不完全性や汚れにより、ビーム断面が真円から楕円に変形する場合がある。これにより、特定の方向（例えばX軸）はシャープであるが、直交する方向（Y軸）はボケているという現象が生じる。これを正確に推定するためには、等方的な特徴抽出機では不十分であり、方向依存性を持つ解析手法が不可欠となる 。   

確率的ノイズ（ショットノイズ）: 高倍率観察や高速スキャンにおいては、単位画素あたりの入射電子数が減少し、ポアソン分布に従うショットノイズが顕著になる。ボケ推定における最大の問題は、高周波成分としての「エッジ」と「ノイズ」の分離である。単純な微分ベースの特徴抽出機はノイズをエッジとして誤検出し、誤った合焦位置（False Peak）を示すリスクがある 。   

1.2 オートフォーカスと画質評価（IQA）の役割
ボケ推定技術は主に二つの用途に応用される。第一に、自動合焦（オートフォーカス）システムにおける「合焦尺度関数（Focus Measure Function）」としての利用である。ここでは、画像からスカラー値を算出し、その値が最大となるレンズ電流値を探索する。理想的な特徴抽出機は、合焦位置で単峰性（Unimodality）を示し、ノイズに対して堅牢である必要がある 。第二に、取得済み画像の品質管理（Quality Control）である。特に半導体検査では、膨大な画像の中からボケた画像を自動的に排除、あるいは再撮像するための判定基準として、参照画像なし（No-Reference）での絶対的な鮮鋭度評価が求められる 。   

2. 空間領域における特徴抽出手法
空間領域（Spatial Domain）の手法は、画素輝度値の配列に対して直接演算を行うものであり、計算コストが低く、リアルタイム性が求められるオートフォーカスシステムにおいて最も広く採用されているアプローチである。

2.1 勾配ベース（微分）演算子
合焦した画像は、ボケた画像と比較して急峻なエッジを持ち、その結果として大きな勾配（Gradient）エネルギーを持つという前提に基づく手法である。

2.1.1 Tenengrad演算子とその派生形
Tenengradは、Sobel演算子を用いて画像の勾配振幅を計算し、その二乗和をとる手法であり、SEMオートフォーカスにおけるベンチマークとして広く認知されている。画像 I(x,y) に対する水平方向勾配 G 
x
​
  と垂直方向勾配 G 
y
​
  をSobelカーネルを用いて算出し、以下の式で定義される。

$$ \text{Tenengrad} = \sum_{x} \sum_{y} (G_x(x,y)^2 + G_y(x,y)^2) \quad \text{for } \sqrt{G_x^2 + G_y^2} > T $$

ここで T はノイズによる微小な勾配を無視するための閾値である。

特性: Tenengradはテクスチャが豊富な試料（例えば金蒸着されたカーボン試料など）において極めて高い感度を示し、合焦位置でのピークが鋭い 。   

課題: 閾値 T の設定がSEMの加速電圧やプローブ電流によるノイズレベルの変化に依存するため、適応的な閾値設定が必要となる。閾値が低すぎるとノイズを信号として積算し、高すぎると微細なテクスチャ情報を切り捨ててしまう 。   

2.1.2 Brenner勾配
Brenner演算子は、隣接画素ではなく、2画素離れた画素との差分をとることで、最小単位の高周波ノイズ（1画素単位の変動）の影響を軽減しつつエッジを検出する手法である。

F 
Brenner
​
 = 
x,y
∑
​
 (I(x+2,y)−I(x,y)) 
2
 
SEMへの適用性: 計算が単純であるため、FPGA等のハードウェア実装が容易である。また、微細なノイズに対してある程度の平滑化効果を持つため、低線量（Low-Dose）イメージングにおいてTenengradよりも安定した挙動を示す場合がある 。   

2.1.3 二階微分（ラプラシアン）とその改良
ラプラシアン（Laplacian）は画像の二階微分 ∇ 
2
 I を利用するもので、エッジのゼロ交差検出などに用いられる。

エネルギー・オブ・ラプラシアン (EOL): ラプラシアン値の二乗和。

修正ラプラシアン (Modified Laplacian - LAPM): 通常のラプラシアンカーネル（例：[0,1,0;1,−4,1;0,1,0]）では、正負の勾配が相殺される可能性があるため、絶対値の和を用いる改良版。

F 
LAPM
​
 = 
x,y
∑
​
 ∣2I(x,y)−I(x−1,y)−I(x+1,y)∣+∣2I(x,y)−I(x,y−1)−I(x,y+1)∣
対角ラプラシアン (LAPD): 対角成分の微分も考慮に入れることで、斜め方向のエッジに対する感度を向上させたもの 。   

限界: 二階微分は高周波成分を強調する特性が強いため、SEM特有のショットノイズを極端に増幅する傾向がある。したがって、高いSN比が確保できる条件下（低速スキャン、高ビーム電流）でのみ有効であるとされる 。   

2.2 統計的特徴量
画像のエッジを明示的に計算するのではなく、画素値の統計分布（ヒストグラム）の変化からボケを推定する手法である。

2.2.1 分散（Variance）および正規化分散
合焦した画像は、輝度のダイナミックレンジを広く使用するため、ヒストグラムの広がり（分散）が大きくなる。一方、ボケた画像は局所的な平均化作用によりヒストグラムが収縮し、分散が低下する。

F 
Variance
​
 = 
MN
1
​
  
x,y
∑
​
 (I(x,y)−μ) 
2
 
ロバスト性: Cornell大学の研究  によれば、分散法はノイズに対して最も堅牢な指標の一つである。勾配法がノイズ成分をプラスのスコアとして加算してしまうのに対し、分散は画像全体の統計量であるため、局所的なノイズスパイクの影響を受けにくい。   

正規化分散: SEMでは電子銃の不安定性により画像の平均輝度 μ が変動することがある。これを補正するため、分散を平均輝度で除算した正規化分散（Coefficient of Variationの二乗に相当）が用いられることが多い 。これにより、明るさの変動による誤ったボケ推定を防ぐことができる。   

2.2.2 自己相関（Autocorrelation）
画像の自己相関関数は、画素間の空間的な相関を表す。鮮鋭な画像では、隣接画素間であってもエッジを挟めば相関が低くなるため、自己相関関数の立ち下がりが急峻になる。

VollahのF4/F5関数: 自己相関に基づく具体的な評価関数であり、特にノイズの多い画像において、勾配法よりも優れた再現性を示すことが報告されている 。   

F 
Auto
​
 = 
x,y
∑
​
 I(x,y)⋅I(x+1,y)− 
x,y
∑
​
 I(x,y)⋅I(x+2,y)
この式は、1画素シフトした相関と2画素シフトした相関の差分をとることで、画像の鮮鋭度成分を抽出している。

3. 周波数領域および変換領域における解析手法
SEM画像、特に半導体パターンのような周期的構造を持つ試料や、極めてノイズの多い試料においては、空間領域での解析よりも周波数領域での解析が有効である。ボケはローパスフィルタとして作用するため、高周波成分の減衰量を定量化することでボケを推定できる。

3.1 高速フーリエ変換（FFT）とパワースペクトル解析
FFTを用いたパワースペクトル解析は、SEMの調整において二つの重要な役割を果たす。

3.1.1 鮮鋭度指標としての利用
画像のパワースペクトル P(u,v) において、高周波領域のエネルギー積分値を合焦尺度とする。

F 
FFT
​
 =∬ 
Ω
​
 ∣P(u,v)∣⋅ 
u 
2
 +v 
2
 

​
 dudv
ここで Ω は高周波領域を示す。この手法は、ノイズの影響を受けやすい高周波端（最高周波数付近）を除外しつつ、中間周波数帯域から高周波数帯域のエネルギーを評価することで、安定したボケ推定が可能である 。   

3.1.2 非点収差の補正とビーム形状推定
SEMにおけるボケの主因の一つである非点収差は、空間領域では方向依存的なボケとして現れるが、FFTスペクトル上ではスペクトルの形状が楕円形に歪む現象として観測される。

原理: 真円のビーム（非点収差なし）で走査された等方的な試料画像のFFTスペクトルは、概ね円形に広がる。しかし、非点収差が存在する場合、ビームが収束している方向の周波数成分は高く保たれるが、発散している方向の成分は減衰するため、スペクトルが楕円形になる 。   

実装: スペクトルの閾値処理を行い、フィッティングした楕円の長軸・短軸比と角度を計算することで、非点補正コイル（Stigmator）へのフィードバック量を算出できる。これにより、フォーカスレンズと非点補正コイルの同時最適化が可能となる。

3.1.3 スペクトルカートシス（Spectral Kurtosis）
パワースペクトルのエネルギー総和だけでなく、その分布形状（尖度：Kurtosis）に着目した手法である。

定義: 多変量カートシスを用いて、周波数領域における係数の分布が正規分布からどれだけ逸脱しているかを測定する 。   

SEMへの利点: ボケた画像は低周波成分が支配的となり、スペクトル分布の裾が狭くなる（尖度が低い）。一方、鮮鋭な画像は高周波まで成分が広がるため、分布の裾が重くなる（尖度が高い）。この指標は「無次元量」であるため、画像のコントラストや照明強度の変化に対して不変（Invariant）であるという極めて強力な特性を持つ。コントラスト変動が激しいSEM画像において、閾値調整なしで安定した評価が可能となる 。   

3.2 離散コサイン変換（DCT）
DCTはJPEG圧縮などの基盤技術であるが、ボケ推定においても計算効率の高い手法として利用される。

ブロックベースのボケ判定: 画像を 8×8 画素のブロックに分割し、各ブロックのDCT係数を計算する。鮮鋭なブロックでは高周波成分（AC係数）が非ゼロの値を持つが、ボケたブロックでは高周波成分が量子化によりゼロになる傾向がある。

ヒストグラム解析: 非ゼロの高周波係数の出現頻度（ヒストグラム）を解析することで、画像全体のボケ度を推定する 。   

中間周波数DCT (MF-DCT): 高周波ノイズの影響を避けるため、DCT係数の中間周波数帯域のみを用いて合焦度を算出する手法。半導体パターンのような周期的構造を持つSEM画像において、SN比の良いボケ推定が可能である 。   

3.3 ウェーブレット変換：DWTとDT-CWT
ウェーブレット変換は、空間情報と周波数情報を同時に扱えるため、局所的なボケの解析に優れている。しかし、通常の離散ウェーブレット変換（DWT）にはSEM画像処理において重大な欠点があることが指摘されている。

3.3.1 通常のDWTの限界
DWT（Haarウェーブレットなど）を用いて画像をLL, LH, HL, HHのサブバンドに分解し、高周波サブバンドのエネルギー比率をボケ指標とする手法がある 。しかし、DWTは「シフト不変性（Shift Invariance）」を持たない。SEMのステージ微動やドリフトにより画像がわずかにシフトしただけで、ウェーブレット係数のエネルギーが大きく変動し、合焦探索においてノイズとなる問題がある 。また、方向選択性が水平・垂直・対角の3方向に限定されるため、任意の角度を持つ半導体パターンの評価には不向きである。   

3.3.2 デュアルツリー複素ウェーブレット変換（DT-CWT）
上記のDWTの欠点を克服するために推奨されるのが、DT-CWT（Dual Tree Complex Wavelet Transform）である 。   

シフト不変性: 実部と虚部を持つ2つのフィルタバンクを並列に用いることで、入力画像のシフトに対して係数の振幅がほぼ一定に保たれる。これにより、振動のある環境下でも安定したフォーカス曲線が得られる。

高い方向分解能: ±15 
∘
 ,±45 
∘
 ,±75 
∘
  の6方向の指向性を持つサブバンドを生成できる。これにより、半導体配線の方向に応じた特異的なボケ（例えば、配線に垂直な方向のボケのみを検出する）が可能となる。

ノイズ除去との統合: DT-CWTはポアソンノイズとエッジ構造の分離能力に優れており、ボケ推定の前処理としてのデノイズ（Denoising）においても、エッジを保存しつつショットノイズを除去する性能がDWTより高い 。   

3.4 チェビシェフモーメント（Chebyshev Moments）
画像のモーメント（Moment）は、画像の形状やテクスチャを記述する大域的な特徴量である。従来の幾何学的モーメントやルジャンドルモーメントは連続関数を前提としており、デジタル画像への適用時に離散化誤差が生じるが、チェビシェフモーメントは離散領域で定義される直交多項式を用いるため、この誤差を排除できる 。   

Shape from Focus (SFF) への応用: 焦点位置を変えながら撮影した一連のSEM画像群（Zスタック）から3次元形状を復元する際、チェビシェフモーメントを用いた合焦尺度が利用される。高次モーメントと低次モーメントのエネルギー比をとることで、単峰性が高く、かつ単調減少する理想的なフォーカス曲線が得られることが報告されている 。   

直交性: 各モーメントが独立しているため情報の重複がなく、少ない特徴量で効率的にボケを記述できる 。   

4. テクスチャ解析および位相ベースの記述子
勾配や周波数エネルギーに依存しない、より人間の視覚特性に近い、あるいは試料の組成コントラストに依存しない特徴抽出手法について述べる。

4.1 輝度共起行列（GLCM: Gray Level Co-occurrence Matrix）
GLCMは、画素の輝度値の空間的な配置関係（テクスチャ）を統計的に解析する手法である。特に、エッジが明確でない生物試料（バイオフィルム等）や多孔質材料のSEM像において威力を発揮する。

原理: ある距離 d と方向 θ だけ離れた2つの画素が、それぞれ輝度 i と j を持つ確率 P(i,j) を行列化する。

ボケに感応する特徴量:

コントラスト (Contrast): ∑(i−j) 
2
 P(i,j)。鮮鋭な画像では輝度差の大きい画素対が多く、値が大きくなる。ボケると低下する。

均質性 (Homogeneity): ∑P(i,j)/(1+∣i−j∣)。ボケた画像では隣接画素が似た値を持つため、値が大きくなる 。   

エントロピー (Entropy): 鮮鋭な画像は情報量（無秩序さ）が多く、ボケると平滑化されエントロピーが低下する 。   

計算コスト: O(N 
2
 ) の計算量を要するため、オートフォーカスの探索ループ内で全画素に対して計算するのは困難であるが、関心領域（ROI）に限定することで実用可能となる 。   

4.2 局所バイナリパターン（LBP: Local Binary Patterns）
LBPは、各画素とその周囲の画素の輝度差を符号化（0または1）し、2進数として表現するテクスチャ記述子である。

ボケ領域のセグメンテーション: 画像内の一部だけがボケている（被写界深度外など）場合、LBPヒストグラムの分布が鮮鋭領域とボケ領域で明確に異なることを利用して、ボケ領域を特定できる 。   

照明変動への不変性: LBPは画素間の「大小関係」のみに依存するため、画像全体の明るさが変動しても（単調増加であれば）パターンは変化しない。これは、帯電によるコントラスト変動やビーム電流のドリフトが発生しやすいSEMにおいて、分散法などよりも安定した指標を提供する 。   

4.3 局所位相コヒーレンス（LPC: Local Phase Coherence）
従来の手法（勾配法、分散法）はすべて「振幅（Amplitude）」や「エネルギー」に依存しているため、画像のコントラストが低いと感度が低下する。これに対し、位相（Phase）に着目した画期的な手法がLPCである。

位相合同性 (Phase Congruency) の理論: 人間の視覚システムは、信号のフーリエ成分の位相が揃う（Congruent）位置を「特徴（エッジや線）」として知覚する。鮮鋭なエッジ部分では、異なる周波数スケールにわたってウェーブレット係数の位相が揃うが、ボケが発生すると位相の関係が崩れる 。   

コントラスト不変性: 位相情報は信号の振幅（コントラスト）に依存しない。したがって、LPCに基づくボケ推定は、組成の違いによりコントラストが大きく異なるSEM画像（例えば、明るい金属粒子と暗いカーボン基板が混在する視野）においても、絶対的な鮮鋭度指標として機能する 。   

LPC-SI (Sharpness Index): 複素ウェーブレット変換領域でスケール間の位相の一貫性を重み付けして統合することで、参照画像なしで高精度な鮮鋭度マップを生成できる 。   

5. 参照画像なし画質評価（NR-IQA）の適用
SEMの現場では、比較対象となる「完全に鮮鋭な原画像」が存在しないため、参照画像なし（No-Reference: NR）で画質を評価する技術が必要となる。

5.1 自然シーン統計（NSS）モデル
「自然な（高品質な）画像は特定の統計的規則に従い、歪み（ボケ）はその規則を乱す」という仮説に基づく手法群である。

BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): 局所的に正規化された輝度係数（MSCN）の統計量を用いる。本来は自然画像用だが、MRIや顕微鏡画像用に再学習させることで、SEM特有の歪みに対応可能であることが示されている 。   

NIQE (Natural Image Quality Evaluator): 「教師なし」アプローチであり、人間の主観スコア付きの学習データを必要としない。高品質なSEM画像群から「理想的な統計モデル」を構築し、それとの距離（逸脱度）を計算する。この手法は、特定の歪みに特化せず、未知の劣化にも対応できるため、SEMの多様な試料に対して柔軟性が高い 。   

PIQE (Perception based Image Quality Evaluator): 学習を一切行わず、ブロックごとの局所的な歪みを推定する手法。エッジやノイズの多いブロックを自動的に識別し、知覚的な品質スコアを算出する。局所的な鮮鋭度マップを出力できるため、SEM画像の不均一なボケ（像面湾曲や傾斜試料によるもの）の解析に適している 。   

5.2 Just Noticeable Blur (JNB) と知覚的閾値
物理的なボケ量と、人間が知覚するボケ感は必ずしも一致しない。JNB（Just Noticeable Blur）は、人間がボケを感知できる最小の閾値を定義する概念である。

CPBD (Cumulative Probability of Blur Detection): 画像内のエッジを検出し、そのエッジ幅がJNB閾値以下である（すなわち十分に鋭い）確率を累積した指標 。   

SEMへの意義: 電子ビームの物理的限界により、SEM画像はある程度のエッジ幅（ボケ）を必然的に持つ。CPBDを用いることで、物理的限界を超えて過剰にフォーカスを探索する（ハンチングする）ことを防ぎ、実用上十分な鮮鋭度に達した時点で探索を終了させるアルゴリズムが可能となる 。   

6. 深層学習およびデータ駆動型アプローチ
従来の特徴抽出機（ハンドクラフト特徴量）は、設計者が想定した劣化モデルには強いが、未知のノイズや複合的な劣化には弱い。近年、深層学習（Deep Learning）を用いたアプローチがSEMのボケ推定と補正に革命をもたらしている。

6.1 CNNベースのオートフォーカスと分類
畳み込みニューラルネットワーク（CNN）は、画像から直接、最適な特徴量を学習する。

AENet と ACNet: SEM専用に設計されたデュアルネットワークシステムが提案されている 。   

AENet (Autofocusing-Evaluation Network): 入力画像の合焦度をスコア（0-9）として絶対評価する。

ACNet (Autofocusing-Control Network): 現在の画像から、「あとどれだけレンズ電流を変化させれば合焦するか」を直接予測する。これにより、従来の山登り法のような反復的な探索を省略し、高速な合焦が可能となる。

WaveCNet (Wavelet Integrated CNN): CNNのプーリング層（ダウンサンプリング）において高周波情報が失われるのを防ぐため、離散ウェーブレット変換（DWT）をプーリングの代わりに用いるアーキテクチャ。これにより、ノイズ耐性が大幅に向上し、SEMのような高ノイズ画像でもロバストな分類が可能となる 。   

6.2 TransformerとAttention機構の導入
自然言語処理で成功したTransformerモデルは、大域的な文脈（Global Context）を捉える能力に優れており、IQA（画質評価）の分野でもSOTA（State-of-the-Art）を記録している。

MANIQA (Multi-dimension Attention Network): Vision Transformer (ViT) をベースに、チャンネル方向と空間方向の両方に対してAttention（注意機構）を適用するモデル 。   

特長: 局所的な特徴（エッジの鋭さ）だけでなく、画像全体の大域的な相互作用を考慮できるため、GANで生成された画像の不自然なテクスチャや、SEMの複雑な帯電アーティファクトによる局所的な歪みを正しく評価できる。

HyperIQA: 画像の「内容（意味）」と「画質」を分離して学習するモデル。Hyper Networkが画像の内容（例：回路パターンか、細胞か）に応じて画質予測ルールを動的に生成する。多種多様なSEM試料に対応する汎用的なボケ推定器として期待される 。   

6.3 自己教師あり学習（SSL）とノイズ除去
SEM画像における最大の課題は、「正解データ（Ground Truth）」の不足である。何千枚もの画像に専門家がボケスコアを付けるのはコストが高い。

Self-Supervised Learning (SSL): ラベルなしの大量のSEM画像を用いて、画像の回転予測やパズル復元などのプレテキストタスク（Pretext Task）を解かせることで、SEM画像特有の構造やテクスチャ表現を学習させる 。こうして事前学習されたモデルは、少数のラベル付きデータでのファインチューニングにより、高精度なボケ推定やセグメンテーションを実現する 。   

Noise2Noise: ノイズ除去における革新的な学習手法。クリーンな画像（正解）を用意する代わりに、「同じシーンの異なるノイズ画像」のペアを用いて学習する 。   

SEMへの適用: SEMでは同一視野を複数回スキャンすることで容易にノイズペアを取得できる。この手法により、ショットノイズを除去し、真の構造的ボケのみを抽出するネットワークを学習させることが可能となり、低線量イメージングにおけるボケ推定精度が飛躍的に向上する。

7. 比較評価と実装戦略
これまでに調査した多種多様な特徴抽出機は、それぞれ異なる特性を持つ。SEMの使用目的や試料の性質に応じて最適な手法を選択する必要がある。

7.1 特性の比較表
以下の表は、主要な特徴抽出手法のSEMにおける特性をまとめたものである。

手法カテゴリ	具体的手法	感度 (Sensitivity)	ノイズ耐性 (Robustness)	計算コスト	特記事項
空間勾配	Tenengrad	高	低 (閾値依存)	中	高コントラスト試料に最適
Brenner	中	中	低	高速スキャン・粗調整向き
統計	Variance	中	高	低	単峰性が高く、安定している
Autocorrelation	中	高	高	周期的なパターンに強い
周波数/変換	FFT/Kurtosis	高	高 (コントラスト不変)	高	非点収差の補正が可能
DT-CWT	高	非常に高い	高	シフト不変性、方向解析に優れる
DCT	中	中	低	ビデオレート処理に適する
テクスチャ/位相	GLCM	低	高	非常に高	生物試料・低エッジ画像向き
LPC (Phase)	高	非常に高い	高	コントラスト変動に影響されない
深層学習	CNN (ACNet)	非常に高	非常に高い	高 (GPU要)	合焦制御を直接予測可能
MANIQA	最高	最高	非常に高	知覚的な画質評価に最適
7.2 シナリオ別推奨戦略
シナリオA：半導体パターンの高速検査・計測 (CD-SEM)
推奨: Tenengrad または DT-CWT

理由: 配線パターンはエッジが明確であり、Tenengradが高い感度を発揮する。また、特定の配線方向（縦・横）に対して感度を持たせる必要がある場合、DT-CWTの方向選択性が有利である。

高速化: 粗調整にはBrennerを用い、最終段の微調整でのみTenengradやDT-CWTを用いる階層的な探索が有効である。

シナリオB：低加速電圧・低ドーズ観察 (Biological/Cryo-SEM)
推奨: LPC (Local Phase Coherence) または Variance

理由: 信号量が少なくノイズが多い環境では、微分ベースの手法は破綻する。LPCはコントラストに依存せず、位相の一貫性を見るため、低コントラストな生物試料のエッジを捉えるのに最適である。計算リソースが限られる場合は、ノイズ耐性の高い分散法（Variance）が次善の策となる。

シナリオC：非点収差を含む精密調整
推奨: FFTパワースペクトル解析

理由: ボケの量だけでなく「方向（非点収差）」を同時に診断できるのはFFTの最大の強みである。スペクトルの楕円形状を解析することで、フォーカスとスティグマ（X/Y）の3軸同時制御が可能となる。

シナリオD：定量的な画質保証・自動選別
推奨: NIQE または MANIQA

理由: 参照画像が存在しない状況で、「この画像は解析に耐えうるか」を判定するには、学習ベースのNR-IQA指標が必要である。NIQEは良品データのみでモデル構築が可能であり、MANIQAは人間の視覚に近い高度な判定を提供する。

8. 結論
本調査により、SEM画像のボケ推定には、単純なエッジ検出から高度な深層学習まで多層的なアプローチが存在することが明らかになった。

古典的手法の堅実性: Variance（分散法） や Brenner といった単純な手法は、その計算効率とノイズ耐性のバランスから、依然としてオートフォーカスの初期段階において強力な選択肢である。

ウェーブレットの優位性: 周波数解析においては、通常のDWTではなく、DT-CWT（デュアルツリー複素ウェーブレット変換） が、シフト不変性と方向選択性の観点からSEMに最も適した変換手法である。

位相情報の重要性: コントラストが変動しやすいSEMにおいて、振幅（輝度）ではなく位相（LPC/Spectral Kurtosis） に着目することで、試料組成に依存しない普遍的な鮮鋭度評価が可能となる。

AIによるパラダイムシフト: Self-Supervised Learning や Transformer (MANIQA) の導入は、単なる「エッジの鋭さ」を超えた、ノイズやアーティファクトを考慮した「知覚的な画質」の評価を可能にしつつある。特に Noise2Noise のような学習戦略は、SEM特有のノイズ問題を解決する鍵となる。

今後の展望として、FPGA等のエッジデバイス上で動作する軽量なCNNモデル（ACNetの軽量版など）の実装や、DT-CWTと深層学習を組み合わせたハイブリッドな特徴抽出器の開発が、次世代の高速・高精度SEMシステムの実現に向けた重要な研究テーマとなるであろう。


researchgate.net
Process flow of auto-focus algorithm. | Download Scientific Diagram - ResearchGate
新しいウィンドウで開く

researchgate.net
Image sharpness measurement in scanning electron microscopy—Part II - ResearchGate
新しいウィンドウで開く

researchgate.net
Image sharpness measurement in scanning electron microscopy—Part I - ResearchGate
新しいウィンドウで開く

researchgate.net
A robust focusing and astigmatism correction method for the scanning electron microscope - ResearchGate
新しいウィンドウで開く

csl.cornell.edu
Sharpness Search Algorithms for Automatic Focusing in the Scanning Electron Microscope
新しいウィンドウで開く

researchgate.net
Sharpness function reaches its optimum at the in-focus image . The goal... - ResearchGate
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Robust autofocusing for scanning electron microscopy based on a dual deep learning network - PMC - NIH
新しいウィンドウで開く

mdpi.com
No-Reference Image Quality Assessment Using the Statistics of Global and Local Image Features - MDPI
新しいウィンドウで開く

opencv.org
Autofocus using OpenCV: A Comparative Study of Focus Measures for Sharpness Assessment
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Quantitative Evaluation of Focus Measure Operators in Optical Microscopy - PMC - NIH
新しいウィンドウで開く

preprints.org
Study on the Design of Quantitative Metrics for Focus Measure Operators - Preprints.org
新しいウィンドウで開く

arxiv.org
[1604.00546] Image Quality Assessment for Performance Evaluation of Focus Measure Operators - arXiv
新しいウィンドウで開く

sites.google.com
Focus Measure - CVIA - Google Sites
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Evaluation of Focus Measures for Hyperspectral Imaging Microscopy Using Principal Component Analysis - PMC - PubMed Central
新しいウィンドウで開く

pubmed.ncbi.nlm.nih.gov
Image sharpness measurement in the scanning electron-microscope--part III - PubMed
新しいウィンドウで開く

researchgate.net
Image sharpness measurement in scanning electron microscope-Part III - ResearchGate
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Time-Varying Spectral Kurtosis: Generalization of Spectral Kurtosis for Local Damage Detection in Rotating Machines under Time-Varying Operating Conditions - PMC - PubMed Central
新しいウィンドウで開く

ijcat.com
Blur Detection Methods for Digital Images-A Survey
新しいウィンドウで開く

en.wikipedia.org
Discrete cosine transform - Wikipedia
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
A Method for Medical Microscopic Images' Sharpness Evaluation Based on NSST and Variance by Combining Time and Frequency Domains - PMC - NIH
新しいウィンドウで開く

openaccess.thecvf.com
Wavelet Integrated CNNs for Noise-Robust Image Classification - CVF Open Access
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Dual tree complex wavelet transform based denoising of optical microscopy images - NIH
新しいウィンドウで開く

cscjournals.org
A DUAL TREE COMPLEX WAVELET TRANSFORM CONSTRUCTION AND ITS APPLICATION TO IMAGE DENOISING - CSC Journals
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Dual tree complex wavelet transform-based signal denoising method exploiting neighbourhood dependencies and goodness-of-fit test - PMC - PubMed Central
新しいウィンドウで開く

researchgate.net
Image reconstruction using Chebyshev moments | Download Scientific Diagram - ResearchGate
新しいウィンドウで開く

ijastems.org
5.Depth Estimation through Combining Chebyshev moments with Bezier-Bernstein polynomial-K.Kanthamma ,Dr.S.A.K.Jilani
新しいウィンドウで開く

semanticscholar.org
[PDF] Image sharpness measure using eigenvalues - Semantic Scholar
新しいウィンドウで開く

naun.org
GLCM based no-reference perceptual blur metric for underwater blur image - NAUN
新しいウィンドウで開く

mecs-press.org
Role of GLCM Features in Identifying Abnormalities in the Retinal Images - MECS Press
新しいウィンドウで開く

arxiv.org
GLCM-Based Feature Combination for Extraction Model Optimization in Object Detection Using Machine Learning - arXiv
新しいウィンドウで開く

medium.com
Feature Extraction of Images using GLCM (Gray Level Cooccurrence Matrix) - Medium
新しいウィンドウで開く

mdpi.com
Blind Remote Sensing Image Deblurring Using Local Binary Pattern Prior - MDPI
新しいウィンドウで開く

cs.usask.ca
LBP-based Segmentation of Defocus Blur - Department of Computer Science | University of Saskatchewan
新しいウィンドウで開く

scispace.com
Defocus image segmentation based on local binary pattern (lbp) and generalized equalization model - SciSpace
新しいウィンドウで開く

scientific.net
A Microscopic Image Sharpness Metric Based on the Local Binary Pattern (LBP)
新しいウィンドウで開く

pubmed.ncbi.nlm.nih.gov
Phase congruency: a low-level image invariant - PubMed
新しいウィンドウで開く

cs.rochester.edu
Image Features from Phase Congruency - Computer Science : University of Rochester
新しいウィンドウで開く

research-repository.uwa.edu.au
Invariant measures of image features from phase information
新しいウィンドウで開く

ece.uwaterloo.ca
Image Sharpness Assessment Based on Local Phase Coherence
新しいウィンドウで開く

researchgate.net
Image Sharpness Assessment Based on Local Phase Coherence | Request PDF
新しいウィンドウで開く

ieeexplore.ieee.org
Image Sharpness Assessment Based on Local Phase Coherence - IEEE Xplore
新しいウィンドウで開く

neuroquantology.com
Image Sharpness Assessment Based On local Phase Coherence | Neuroquantology
新しいウィンドウで開く

pubmed.ncbi.nlm.nih.gov
Modified-BRISQUE as no reference image quality assessment for structural MR images - PubMed
新しいウィンドウで開く

live.ece.utexas.edu
No-Reference Image Quality Assessment in the Spatial Domain
新しいウィンドウで開く

mathworks.com
Image Quality Metrics - MATLAB & Simulink - MathWorks
新しいウィンドウで開く

github.com
No-reference Image Quality Assessment(NIQA) Algorithms (BRISQUE, NIQE, PIQE, RankIQA, MetaIQA) - GitHub
新しいウィンドウで開く

mathworks.com
piqe - Perception based Image Quality Evaluator (PIQE) no-reference image quality score - MATLAB - MathWorks
新しいウィンドウで開く

quality.nfdi4ing.de
Perception based Image Quality Evaluator
新しいウィンドウで開く

scribd.com
Just Noticeable Blur | PDF | Contrast (Vision) | Optical Resolution - Scribd
新しいウィンドウで開く

researchgate.net
A No-Reference Objective Image Sharpness Metric Based on the Notion of Just Noticeable Blur (JNB) | Request PDF - ResearchGate
新しいウィンドウで開く

acorn.stanford.edu
Image Blur Metrics - Stanford University
新しいウィンドウで開く

researchgate.net
MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment
新しいウィンドウで開く

openaccess.thecvf.com
MANIQA: Multi-Dimension Attention Network for No-Reference Image Quality Assessment - CVF Open Access
新しいウィンドウで開く

openaccess.thecvf.com
Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network - CVF Open Access
新しいウィンドウで開く

pmc.ncbi.nlm.nih.gov
Exploring self-supervised learning biases for microscopy image representation - PMC
新しいウィンドウで開く

openaccess.thecvf.com
Self-Supervised Learning with Generative Adversarial Networks for Electron Microscopy - CVF Open Access
新しいウィンドウで開く

researchgate.net
Leveraging unlabeled SEM datasets with self-supervised learning for enhanced particle segmentation - ResearchGate
新しいウィンドウで開く

taylorfrancis.com
Self-Supervised Learning-Based Classification of Scanning Electron Microscope Images of Biofilms - Taylor & Francis eBooks
新しいウィンドウで開く

academic.oup.com
Efficient and Robust SEM Image Denoising for Wafer Defect Inspection - Oxford Academic