"""
EXP017: Qwen3-VL LoRA Fine-tuning for Defocus Estimation
v2 + v5 のsampleデータを結合して学習

EXP016をベースに、v2とv5のsampleデータを結合して学習データを増やす。

使用方法:
```
python EXP/EXP017/train_infer.py
```

必要なパッケージ:
- torch
- transformers
- peft
- bitsandbytes
- albumentations
- PIL
- sklearn
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import random
from sklearn.model_selection import GroupKFold
import albumentations as A
import shutil
import pandas as pd


# ============================================
# パス設定
# ============================================
DATA_DIR_V2 = Path("input/DeepFocusChallenge_v2")
DATA_DIR_V5 = Path("input/DeepFocusChallenge_v5")
OUTPUT_DIR = Path("EXP/EXP017/outputs")
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 定数・設定
# ============================================

TTA_TRANSFORMS = [
    lambda img: img,
    lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    lambda img: img.transpose(Image.TRANSPOSE),
]

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Language Model Attention
    "gate_proj", "up_proj", "down_proj",      # Language Model MLP
    "qkv", "proj",                            # Vision Encoder Attention
    "linear_fc1", "linear_fc2",               # Vision Encoder MLP
]


class Config:
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    num_epochs = 15
    batch_size = 3
    learning_rate = 1e-4
    classifier_lr = 1e-3
    gradient_accumulation_steps = 3
    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.05
    max_image_size = 512
    crop_size = 512
    prompt = "Please determine if this SEM image is out of focus."
    n_folds = 5
    seed = 42
    num_snapshots = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = "exp017"


# ============================================
# Seed固定
# ============================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# ============================================
# 前処理・データセット
# ============================================

def apply_fov_and_crop(image: Image.Image, fov: float, transforms, max_size: int):
    """FOVに基づいてスケーリングし、クロップを適用"""
    scale = fov / 2.0
    image = image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.LANCZOS
    )
    if transforms:
        image = Image.fromarray(transforms(image=np.array(image))["image"])
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS
        )
    return image


def get_train_transforms(crop_size: int):
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ShiftScaleRotate(shift_limit=(-0.15, 0.15), scale_limit=0, rotate_limit=0, p=1.0),
        A.CenterCrop(height=crop_size, width=crop_size),
    ])


def get_val_transforms(crop_size: int):
    return A.Compose([A.CenterCrop(height=crop_size, width=crop_size)])


class TrainDataset(Dataset):
    def __init__(self, data: list, config: Config, transforms=None):
        self.data = data
        self.config = config
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = apply_fov_and_crop(
            image, item["fov"], self.transforms, self.config.max_image_size
        )
        return {
            "image": image,
            "label": item["label"],
            "original_label": item["label"]
        }


class TTADataset(Dataset):
    def __init__(self, data: list, config: Config):
        self.data = data
        self.config = config
        self.val_trans = get_val_transforms(config.crop_size)

    def __len__(self):
        return len(self.data) * 4

    def __getitem__(self, idx):
        img_idx, tta_idx = divmod(idx, 4)
        item = self.data[img_idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = apply_fov_and_crop(
            image, item["fov"], self.val_trans, self.config.max_image_size
        )
        image = TTA_TRANSFORMS[tta_idx](image)
        return {
            "image": image,
            "label": item["label"],
            "original_label": item["label"],
            "image_idx": img_idx,
            "tta_idx": tta_idx
        }


# ============================================
# モデル
# ============================================

class Qwen3VLClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, dropout=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, pixel_values=None,
                image_grid_thw=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        last_token_idx = attention_mask.sum(dim=1) - 1
        pooled = outputs.hidden_states[-1][
            torch.arange(input_ids.shape[0], device=input_ids.device),
            last_token_idx
        ]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits.squeeze(-1), labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.base_model.save_pretrained(path)
        torch.save({
            "classifier": self.classifier.state_dict(),
            "dropout": self.dropout.p
        }, os.path.join(path, "classifier_head.pt"))


# ============================================
# ユーティリティ
# ============================================

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


def collate_fn(batch, processor, prompt):
    images = [item["image"] for item in batch]
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[prompt_text] * len(batch),
        images=images,
        padding=True,
        return_tensors="pt"
    )
    inputs["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float)
    inputs["original_labels"] = torch.tensor(
        [item["original_label"] for item in batch], dtype=torch.float
    )
    if "image_idx" in batch[0]:
        inputs["image_indices"] = torch.tensor(
            [item["image_idx"] for item in batch], dtype=torch.long
        )
        inputs["tta_indices"] = torch.tensor(
            [item["tta_idx"] for item in batch], dtype=torch.long
        )
    return inputs


def load_trained_model(path, config):
    """学習済みモデルをロード"""
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True
    )
    proc = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token

    base = PeftModel.from_pretrained(base, path)
    st = torch.load(os.path.join(path, "classifier_head.pt"), map_location=config.device)
    model = Qwen3VLClassifier(base, base.config.text_config.hidden_size, st["dropout"])
    model.classifier.load_state_dict(st["classifier"])
    return model.to(config.device, dtype=torch.bfloat16), proc


# ============================================
# 評価
# ============================================

class SimpleValDataset(Dataset):
    """TTAなしのシンプルな評価用データセット"""
    def __init__(self, data: list, config: Config):
        self.data = data
        self.config = config
        self.val_trans = get_val_transforms(config.crop_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image = apply_fov_and_crop(
            image, item["fov"], self.val_trans, self.config.max_image_size
        )
        return {
            "image": image,
            "label": item.get("label", 0.0),
            "original_label": item.get("label", 0.0)
        }


class TestDataset(Dataset):
    """テスト推論用データセット"""
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.val_trans = get_val_transforms(config.crop_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.filepath).convert("RGB")
        image = apply_fov_and_crop(
            image, row.FOV, self.val_trans, self.config.max_image_size
        )
        return {
            "image": image,
            "label": 0.0,
            "original_label": 0.0
        }


def evaluate_simple(model, processor, val_data: list, config: Config):
    """TTAなしのシンプル評価（高速）"""
    model.eval()
    loader = DataLoader(
        SimpleValDataset(val_data, config),
        batch_size=config.batch_size * 4,  # TTAがないので大きめバッチ
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, processor, config.prompt)
    )
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = torch.clamp(
                model(**batch)["logits"].squeeze(-1), 0, 100
            ).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["original_labels"].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    rmse = np.sqrt(np.mean((all_labels - all_preds) ** 2))
    return rmse, all_preds, all_labels


def evaluate_with_tta(model, processor, val_data: list, config: Config):
    """TTA付きで評価（テスト推論用）"""
    model.eval()
    loader = DataLoader(
        TTADataset(val_data, config),
        batch_size=config.batch_size * 4,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, processor, config.prompt)
    )
    all_tta_preds = np.zeros((len(val_data), 4))
    all_labels = np.zeros(len(val_data))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = torch.clamp(
                model(**batch)["logits"].squeeze(-1), 0, 100
            ).float().cpu().numpy()

            for i, (img_i, tta_i) in enumerate(zip(
                batch["image_indices"].cpu().numpy(),
                batch["tta_indices"].cpu().numpy()
            )):
                all_tta_preds[img_i, tta_i] = preds[i]
                all_labels[img_i] = batch["original_labels"][i].item()

    final_preds = all_tta_preds.mean(axis=1)
    rmse = np.sqrt(np.mean((all_labels - final_preds) ** 2))
    return rmse, final_preds, all_labels


class CollateFn:
    """pickle可能なcollate関数（Windows対応）"""
    def __init__(self, processor, prompt):
        self.processor = processor
        self.prompt = prompt

    def __call__(self, batch):
        return collate_fn(batch, self.processor, self.prompt)


def predict_test(model, processor, test_df: pd.DataFrame, config: Config):
    """テスト推論（DataLoader使用で高速化）"""
    model.eval()
    loader = DataLoader(
        TestDataset(test_df, config),
        batch_size=config.batch_size * 4,  # 大きめバッチ
        shuffle=False,
        num_workers=0,  # Windows互換性のため0に設定
        collate_fn=CollateFn(processor, config.prompt)
    )
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            preds = torch.clamp(
                model(**batch)["logits"].squeeze(-1), 0, 100
            ).float().cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# ============================================
# 学習
# ============================================

def train_fold(config, train_data, val_data, fold):
    print(f"\n{'='*20} Fold {fold} Start {'='*20}")
    set_seed(config.seed)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES
    )
    model = Qwen3VLClassifier(
        get_peft_model(base_model, lora_config),
        base_model.config.text_config.hidden_size
    ).to(config.device)

    train_loader = DataLoader(
        TrainDataset(train_data, config, get_train_transforms(config.crop_size)),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor, config.prompt),
        worker_init_fn=worker_init_fn
    )

    optimizer = torch.optim.AdamW([
        {"params": model.base_model.parameters(), "lr": config.learning_rate},
        {"params": model.classifier.parameters(), "lr": config.classifier_lr}
    ], weight_decay=0.01)

    out_dir = MODEL_DIR / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(pbar):
            batch = {
                k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            loss = model(**batch)["loss"] / config.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({
                    "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                    "step": global_step
                })

        avg_loss = epoch_loss / len(train_loader)
        rmse_val, _, _ = evaluate_with_tta(model, processor, val_data, config)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Val RMSE: {rmse_val:.4f}")

        # Snapshot保存
        if len(snapshots) < config.num_snapshots or rmse_val < max(s[0] for s in snapshots):
            temp_path = out_dir / f"tmp_ep{epoch+1}"
            model.save_pretrained(str(temp_path))
            snapshots.append((rmse_val, epoch + 1, str(temp_path)))
            snapshots.sort(key=lambda x: x[0])
            if len(snapshots) > config.num_snapshots:
                _, _, p_del = snapshots.pop(-1)
                if os.path.exists(p_del):
                    shutil.rmtree(p_del)
            print(f"Snapshot saved! Best RMSE: {snapshots[0][0]:.4f}")

    # OOF予測（モデル再ロード不要、best snapshotのweightsを現在のモデルにロード）
    print(f"Generating OOF predictions for Fold {fold}...")
    best_rmse, best_ep, best_path = snapshots[0]

    # best snapshotのweightsをロード（ベースモデルは再ロード不要）
    from safetensors.torch import load_file
    adapter_path = os.path.join(best_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        best_lora_state = load_file(adapter_path)
    else:
        best_lora_state = torch.load(os.path.join(best_path, "adapter_model.bin"), map_location=config.device)
    model.base_model.load_state_dict(best_lora_state, strict=False)
    best_cls_state = torch.load(os.path.join(best_path, "classifier_head.pt"), map_location=config.device)
    model.classifier.load_state_dict(best_cls_state["classifier"])

    _, oof_pred, _ = evaluate_simple(model, processor, val_data, config)

    # Snapshot整理
    final_paths = []
    print(f"\nFinalizing Fold {fold} snapshots...")
    for i, (r, ep, p) in enumerate(snapshots):
        name = "snapshot_best" if i == 0 else "snapshot_2nd_best"
        fpath = out_dir / name
        if fpath.exists():
            shutil.rmtree(fpath)
        shutil.move(p, str(fpath))
        processor.save_pretrained(str(fpath))
        final_paths.append(str(fpath))
        print(f"  - {name}: Epoch {ep}, RMSE {r:.4f} -> {fpath}")

    # メモリ解放
    del model
    torch.cuda.empty_cache()

    return final_paths, best_rmse, oof_pred


# ============================================
# 推論
# ============================================

def predict_batch(model, processor, image_paths, fovs, config, use_tta=False):
    """バッチ推論（TTAオプション）"""
    model.eval()
    val_trans = get_val_transforms(config.crop_size)
    all_imgs = []

    for path, fov in zip(image_paths, fovs):
        img = Image.open(path).convert("RGB")
        img = apply_fov_and_crop(img, fov, val_trans, config.max_image_size)
        if use_tta:
            all_imgs.extend(fn(img) for fn in TTA_TRANSFORMS)
        else:
            all_imgs.append(img)

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": config.prompt}]}],
        tokenize=False,
        add_generation_prompt=False
    )
    inputs = processor(
        text=[prompt_text] * len(all_imgs),
        images=all_imgs,
        padding=True,
        return_tensors="pt"
    ).to(config.device)

    with torch.no_grad():
        preds = torch.clamp(model(**inputs)["logits"].squeeze(-1), 0, 100).float()
        if use_tta:
            preds = preds.view(len(image_paths), 4).mean(dim=1)
    return preds.cpu().numpy()


# ============================================
# メイン処理
# ============================================

def main():
    print("=" * 60)
    print("EXP017: Qwen3-VL LoRA Fine-tuning (v2 + v5 combined)")
    print("=" * 60)

    config = Config()
    set_seed(config.seed)

    # ============================================
    # データ読み込み（v2 + v5を結合）
    # ============================================
    print(f"\nLoading data from v2 and v5...")

    # v2 sample data
    sample_v2 = pd.read_csv(DATA_DIR_V2 / "sample.csv")
    sample_v2["filepath"] = sample_v2["filepath"].apply(
        lambda x: str(DATA_DIR_V2 / x.lstrip("./"))
    )
    sample_v2["source"] = "v2"
    print(f"  v2 sample: {len(sample_v2)} images, {sample_v2['pattern'].nunique()} patterns")

    # v5 sample data
    sample_v5 = pd.read_csv(DATA_DIR_V5 / "sample.csv")
    sample_v5["filepath"] = sample_v5["filepath"].apply(
        lambda x: str(DATA_DIR_V5 / x.lstrip("./"))
    )
    sample_v5["source"] = "v5"
    print(f"  v5 sample: {len(sample_v5)} images, {sample_v5['pattern'].nunique()} patterns")

    # v5 test data (推論用)
    test_df = pd.read_csv(DATA_DIR_V5 / "test.csv")
    test_df["filepath"] = test_df["filepath"].apply(
        lambda x: str(DATA_DIR_V5 / x.lstrip("./"))
    )
    print(f"  v5 test: {len(test_df)} images")

    # パターン番号の衝突を避けるため、v2のpatternにオフセットを追加
    max_pattern_v5 = sample_v5["pattern"].max()
    sample_v2["pattern"] = sample_v2["pattern"] + max_pattern_v5 + 1

    # 結合
    train_df = pd.concat([sample_v2, sample_v5], ignore_index=True)
    print(f"\nCombined training data: {len(train_df)} images, {train_df['pattern'].nunique()} patterns")

    # 学習データ準備
    all_data = [
        {
            "image_path": row.filepath,
            "label": float(row.abs_focus),
            "fov": float(row.FOV),
            "pattern": int(row.pattern),
            "source": row.source
        }
        for row in train_df.itertuples()
    ]

    # GroupKFold CV (patternでグループ化してリーク防止)
    groups = [d['pattern'] for d in all_data]
    n_patterns = len(set(groups))
    n_folds = min(config.n_folds, n_patterns)
    gkfold = GroupKFold(n_splits=n_folds)
    oof_preds = np.zeros(len(all_data))
    cv_scores = []

    print(f"\nStarting {n_folds}-Fold GroupKFold CV (grouped by pattern)...")
    print(f"Number of patterns: {n_patterns}")

    for fold, (t_idx, v_idx) in enumerate(gkfold.split(range(len(all_data)), groups=groups)):
        t_data = [all_data[i] for i in t_idx]
        v_data = [all_data[i] for i in v_idx]
        val_patterns = sorted(set([all_data[i]['pattern'] for i in v_idx]))
        val_sources = set([all_data[i]['source'] for i in v_idx])
        print(f"\nFold {fold+1}: Val patterns = {val_patterns}, sources = {val_sources}")
        snapshot_paths, best_rmse, oof_pred = train_fold(config, t_data, v_data, fold + 1)
        cv_scores.append(best_rmse)
        oof_preds[v_idx] = oof_pred

    # CVサマリー
    labels = np.array([d['label'] for d in all_data])
    overall_rmse = np.sqrt(np.mean((labels - oof_preds) ** 2))

    print(f"\n{'='*20} CV Summary {'='*20}")
    for i, s in enumerate(cv_scores):
        print(f"Fold {i+1}: RMSE = {s:.4f}")
    print(f"Mean RMSE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    print(f"Overall OOF RMSE (Ensemble): {overall_rmse:.4f}")

    # OOF保存
    train_df["oof"] = oof_preds
    oof_path = OUTPUT_DIR / f"oof_{config.exp_name}.csv"
    train_df[["id", "oof", "source"]].to_csv(oof_path, index=False)
    print(f"OOF saved: {oof_path}")

    # テスト推論
    print(f"\n{'='*20} Test Inference Start {'='*20}")
    test_results = []

    # ベースモデルを1回だけロード（量子化は1回のみ）
    print("Loading base model (one-time quantization)...")
    from safetensors.torch import load_file
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 最初のsnapshotでLoRAを適用してモデル構築
    first_path = None
    for fold in range(1, n_folds + 1):
        fdir = MODEL_DIR / f"fold_{fold}"
        for name in ["snapshot_best", "snapshot_2nd_best"]:
            path = fdir / name
            if path.exists():
                first_path = str(path)
                break
        if first_path:
            break

    base_model = PeftModel.from_pretrained(base_model, first_path)
    cls_state = torch.load(os.path.join(first_path, "classifier_head.pt"), map_location=config.device)
    model = Qwen3VLClassifier(base_model, base_model.config.text_config.hidden_size, cls_state["dropout"])
    model.classifier.load_state_dict(cls_state["classifier"])
    model = model.to(config.device, dtype=torch.bfloat16)

    # 各snapshotでLoRAアダプターのweightsだけ差し替えて推論
    for fold in range(1, n_folds + 1):
        fdir = MODEL_DIR / f"fold_{fold}"
        for name in ["snapshot_best", "snapshot_2nd_best"]:
            path = fdir / name
            if not path.exists():
                continue
            print(f"Swapping adapter: {path}...")

            # LoRAアダプターのweightsをロード
            adapter_path = os.path.join(str(path), "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                lora_state = load_file(adapter_path)
            else:
                lora_state = torch.load(os.path.join(str(path), "adapter_model.bin"), map_location=config.device)
            model.base_model.load_state_dict(lora_state, strict=False)

            # Classifier headのweightsをロード
            cls_state = torch.load(os.path.join(str(path), "classifier_head.pt"), map_location=config.device)
            model.classifier.load_state_dict(cls_state["classifier"])

            # DataLoaderを使った高速推論
            preds = predict_test(model, processor, test_df, config)
            test_results.append(preds)

    del model
    torch.cuda.empty_cache()

    # 提出ファイル作成
    test_df["abs_focus"] = np.mean(test_results, axis=0)
    submission_path = OUTPUT_DIR / "submission.csv"
    test_df[["id", "abs_focus"]].to_csv(submission_path, index=False)

    print(f"\n{'='*60}")
    print(f"Submission saved: {submission_path}")
    print(f"{'='*60}")

    # 予測統計
    print(f"\nPrediction statistics:")
    print(f"  Mean: {test_df['abs_focus'].mean():.2f}")
    print(f"  Std:  {test_df['abs_focus'].std():.2f}")
    print(f"  Min:  {test_df['abs_focus'].min():.2f}")
    print(f"  Max:  {test_df['abs_focus'].max():.2f}")

    print("\nAll processes completed!")


def debug_inference_speed():
    """推論速度をテストするデバッグモード"""
    import time

    print("=" * 60)
    print("DEBUG: Inference Speed Test")
    print("=" * 60)

    config = Config()
    set_seed(config.seed)

    # テストデータ読み込み（v5）
    test_df = pd.read_csv(DATA_DIR_V5 / "test.csv")
    test_df["filepath"] = test_df["filepath"].apply(
        lambda x: str(DATA_DIR_V5 / x.lstrip("./"))
    )
    print(f"Test data: {len(test_df)} images")

    # 少数のデータでテスト
    n_test = min(50, len(test_df))
    test_df_small = test_df.head(n_test).copy()
    print(f"Using {n_test} images for speed test")

    # モデルパスを探す
    model_path = None
    for fold in range(1, 6):
        fdir = MODEL_DIR / f"fold_{fold}"
        path = fdir / "snapshot_best"
        if path.exists():
            model_path = str(path)
            break

    if model_path is None:
        print("ERROR: No trained model found. Run training first.")
        return

    print(f"\nLoading model from: {model_path}")

    # モデルロード時間計測
    t0 = time.time()
    model, processor = load_trained_model(model_path, config)
    t_load = time.time() - t0
    print(f"Model load time: {t_load:.2f}s")

    # ウォームアップ（最初の推論は遅い）
    print("\nWarmup run...")
    warmup_df = test_df_small.head(5).copy()
    _ = predict_test(model, processor, warmup_df, config)

    # 推論速度計測
    print(f"\nRunning inference on {n_test} images...")
    t0 = time.time()
    preds = predict_test(model, processor, test_df_small, config)
    t_infer = time.time() - t0

    print(f"\n{'='*40}")
    print(f"Results:")
    print(f"  Total time: {t_infer:.2f}s")
    print(f"  Time per image: {t_infer/n_test*1000:.1f}ms")
    print(f"  Images per second: {n_test/t_infer:.2f}")
    print(f"  Predictions: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}")
    print(f"{'='*40}")

    # フルテストの推定時間
    full_time_per_model = t_infer / n_test * len(test_df)
    n_models = 10  # 5 folds × 2 snapshots
    total_estimated = full_time_per_model * n_models
    print(f"\nEstimated full inference time:")
    print(f"  Per model ({len(test_df)} images): {full_time_per_model/60:.1f} min")
    print(f"  Total ({n_models} models): {total_estimated/60:.1f} min")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        debug_inference_speed()
    else:
        main()
