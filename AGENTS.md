# Repository Guidelines

## Project Structure & Module Organization
- `EXP/EXPxxx`: Each experiment folder contains `infer.py`, optional `config/*.yaml`, and writes predictions to `EXP/EXPxxx/outputs` (gitignored). Start from the closest prior EXP to keep feature definitions consistent.
- `input/DeepFocusChallenge_v2`: Local dataset root for `train/`, `test/`, and `sample/`; never commit data.
- `notebooks/DEPTH`: Exploratory notebooks and `mask.py` utilities for vegetation/depth analysis.
- `Docs/research`: Research notes and design logs; update here when adding methodological findings.

## Setup & Dependencies
- Use Python 3.10+; install runtime libs used across experiments: `numpy`, `pandas`, `opencv-python`, `lightgbm`, `scikit-learn`, `tqdm`, `matplotlib`; `torch` is needed only for depth-anything tests.
- Keep environment files out of git (`.env`, `kaggle.json` are ignored). Place large artifacts under `EXP/<id>/outputs` or a local `outputs_*/` directory.

## Build, Test, and Development Commands
- Run an experiment end-to-end (expects `input/DeepFocusChallenge_v2` to exist):
  - `python EXP/EXP003/infer.py` (LightGBM + DeepResearch features)
  - `python EXP/EXP001/infer.py` (baseline LightGBM)
  - `python EXP/EXP000/infer.py` (ridge + minimal blur features)
- Quick feature sanity check on a sample image:
  - `python EXP/EXP000/test_blur_features.py --image_path input/DeepFocusChallenge_v2/sample/020000.JPG`
- Depth demo (heavy, optional): follow `EXP/EXP000/test_depth.py` docstring to install Depth Anything v3, then run with `--image_path ...`.

## Coding Style & Naming Conventions
- Follow PEP8-ish style: 4-space indents; `snake_case` for functions/variables; uppercase constants (`DATA_DIR`, `OUTPUT_DIR`). Type hints are encouraged for function signatures.
- Keep feature extraction in pure helper functions returning dicts; separate dataset I/O, CV loops, and plotting. Prefer `pathlib.Path` over raw strings for paths.
- Store tunable params in YAML under `EXP/*/config` and avoid duplicating hardcoded paths.

## Testing Guidelines
- Before long runs, validate logic on a small subset or single fold; confirm feature stats match expectations (e.g., blur features decrease as images defocus).
- For visualization scripts, verify plots save under `EXP/<id>/outputs` and keep Matplotlib interactive usage optional.
- Log seeds, CV splits, and output filenames in console prints to make reruns reproducible; share any new metrics in `EXP/EXP_SUMMARY.md` if applicable.

## Commit & Pull Request Guidelines
- Existing history uses concise experiment tags (`EXP012`, `EXP010`). Keep the prefix and add a short description when helpful: `EXP013: add contrast-invariant features`.
- PRs should include: goal/approach, data subset used, key metrics (CV/LB), commands run, and pointers to new configs or output locations. Attach small screenshots/plots when they aid review.
- Do not commit datasets, generated CSVs, model weights, or notebook outputs; ensure git status stays clean after a full run.

## Data & Security
- Respect `.gitignore`: keep `input/`, `outputs/`, and model artifacts local. If sharing results, export paths only, not files.
- Avoid embedding credentials in scripts or logs; prefer environment variables and keep `.env` files local.

## Test Data Handling
- テスト画像の特徴量は各画像単体からのみ計算し、他のテスト画像の情報を混ぜない（クラスタリングや全体統計の作成は禁止）。
- 逐次推論が原則。バッチ推論は可だが、各サンプルの推論に他サンプルの特徴を利用しないこと。
- テスト特徴量を使った集約・正規化・補正を行わない（例: test全体の平均や分散でスケーリングしない）。


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLCaT Deep Focus Challenge - A Kaggle-style competition to estimate defocus amount (depth of field deviation) in Scanning Electron Microscope (SEM) images. The goal is building a domain-robust regression model that handles varying imaging conditions.

**Evaluation Metric**: Pattern-averaged RMSE (macro-averaged RMSE across image patterns)

## Project Structure

```
input/DeepFocusChallenge_v2/   # Competition data
├── train.csv, test.csv        # Metadata with imaging parameters
├── sample.csv                 # Labeled reference data (55 images, focus 0-80)
├── train/, test/, sample/     # Image directories (JPG)
└── sample_submission.csv      # Submission format

notebooks/DEPTH/               # Model experiments
├── depth.ipynb                # ZoeDepth exploration
├── depthanything3.ipynb       # Depth Anything v3 experiments
└── mask.py                    # Image analysis utilities
```

## Key Data Characteristics

- **Training data**: Mostly best-focus images (abs_focus=0) - synthetic blur generation required
- **Domain shift challenge**: Train/test have different device conditions (FOV, Beam Rotation, HV, Beam Current, VacuumMode)
- **Target variable**: `abs_focus` (defocus amount in micrometers)

## Development Environment

- Primary workflow: Jupyter notebooks (Google Colab compatible)
- Dependencies installed via pip in notebooks
- Key packages: `timm==0.6.7`, PyTorch with CUDA, OpenCV, matplotlib

## Experiment Structure

実験は `EXP/{EXPName}/` に配置する。

```
EXP/EXP001/
├── train.py
├── infer.py
└── config/
    ├── child-exp000.yaml
    └── child-exp001.yaml
```

### Versioning Rules

- **パラメータ変更のみ**: child をリビジョンアップ (例: child-exp000.yaml → child-exp001.yaml)
- **Pipeline変更**: EXP側をリビジョンアップ (例: EXP001 → EXP002)
- **コード履歴**: 過去のコンテキストを残すため、コードは削除せず増やしていく

### Experiment Log

実験を行ったら `EXP/EXP_SUMMARY.md` を都度更新すること。記載内容:
- なぜその実験を行ったのか（目的・仮説）
- どのような結果が得られたのか（スコア・知見）

## Technical Approach Hints

The competition emphasizes domain robustness. Key approaches to consider:
- **Synthetic blur simulation**: Apply Gaussian blur or PSF to best-focus images for training data augmentation
- **Reference calibration**: Use sample/ data (has labels) for model calibration
- **Domain adaptation**: Models should not overfit to specific imaging conditions
