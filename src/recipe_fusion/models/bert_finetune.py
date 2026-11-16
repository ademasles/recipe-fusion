"""Fine-tuned BERT text reference (sheet B4, method ``text_bert_ft``).

bert-base-uncased fully unfrozen, with the masked mean of ``last_hidden_state`` used in B1, then
Dropout(0.1) -> Linear(768, 101). Same preprocessing and sequence length L as the B1 feature
cache. Recipe of the BERT paper (Devlin et al., 2019, appendix A.3), an assumed exception to
protocol 2.4: AdamW, lr in {2e-5, 3e-5, 5e-5} chosen on the validation top-1 with seed 0, weight
decay 0.01 except on biases and LayerNorm, linear warmup over 10 % of the steps then linear decay
to 0, batch 32, at most 4 epochs, best validation-loss epoch kept, bfloat16 autocast on GPU
(done by ``training.fit``). Validation and test predictions are written for 3 seeds. No test
metric is computed here.
"""

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
import transformers
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, BertModel

from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import labels_for, read_features
from recipe_fusion.features.text import (
    PREPROCESSING,
    git_commit,
    masked_mean,
    preprocess,
    resolve_revision,
)
from recipe_fusion.models.pooled_mlp import best_config
from recipe_fusion.preds import ckpt_path, preds_path, write_preds
from recipe_fusion.seed import pick_device, seed_everything
from recipe_fusion.training import FitResult, TrainConfig, fit, predict

log = logging.getLogger(__name__)


class BertClassifier(nn.Module):
    """BERT encoder, masked mean of the last hidden states, Dropout, Linear."""

    def __init__(self, encoder: BertModel, n_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, n_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = masked_mean(hidden.float(), attention_mask)  # same pooling as B1, in float32
        return self.classifier(self.dropout(pooled))


def save_classifier(directory: Path, model: BertClassifier, **extra: float | int | str) -> None:
    """Encoder in the Hugging Face format plus the classifier layer, in one directory."""
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory)
    torch.save(
        {
            "state_dict": model.classifier.state_dict(),
            "n_classes": model.classifier.out_features,
            "dropout": model.dropout.p,
            "extra": extra,
        },
        directory / "classifier.pt",
    )


def load_classifier(directory: Path, device: str | torch.device = "cpu") -> BertClassifier:
    checkpoint = torch.load(directory / "classifier.pt", map_location=device, weights_only=True)
    encoder = BertModel.from_pretrained(directory, add_pooling_layer=False)
    model = BertClassifier(encoder, checkpoint["n_classes"], checkpoint["dropout"])
    model.classifier.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


class TokenBatches:
    """Rows of tokenized titles, served as ``{"input_ids", "attention_mask"}`` int64 tensors."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def get(self, rows: npt.NDArray[np.int64]) -> dict[str, Tensor]:
        return {
            "input_ids": torch.from_numpy(np.ascontiguousarray(self.input_ids[rows])).long(),
            "attention_mask": torch.from_numpy(
                np.ascontiguousarray(self.attention_mask[rows])
            ).long(),
        }


@dataclass(frozen=True)
class TokenData:
    classes: list[str]
    max_len: int
    train: TokenBatches
    y_train: np.ndarray
    val: TokenBatches
    y_val: np.ndarray
    ids_val: list[str]
    test: TokenBatches
    ids_test: list[str]


def tokenize(texts: Sequence[str], tokenizer, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Input ids and attention mask padded or truncated to ``max_len`` ([CLS], [SEP] included)."""
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int8)


def cache_max_len(text_dir: Path) -> int:
    """L of the B1 cache, after checking it used the same preprocessing."""
    meta = read_features(text_dir, "train").meta
    if meta.get("preprocessing") != PREPROCESSING:
        raise ValueError(
            f"text cache preprocessing is {meta.get('preprocessing')!r}, expected {PREPROCESSING!r}"
        )
    if "L" not in meta:
        raise ValueError(f"text cache meta in {text_dir} has no L")
    return int(meta["L"])


def load_token_data(splits: Splits, tokenizer, max_len: int) -> TokenData:
    def batches(samples: Sequence[Sample]) -> TokenBatches:
        return TokenBatches(*tokenize([preprocess(s.text) for s in samples], tokenizer, max_len))

    ids_train = [s.image_name for s in splits.train]
    ids_val = [s.image_name for s in splits.val]
    return TokenData(
        classes=splits.classes,
        max_len=max_len,
        train=batches(splits.train),
        y_train=labels_for(ids_train, splits.train_all, splits.classes),
        val=batches(splits.val),
        y_val=labels_for(ids_val, splits.train_all, splits.classes),
        ids_val=ids_val,
        test=batches(splits.test),
        ids_test=[s.image_name for s in splits.test],
    )


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on matrices only: biases and LayerNorm parameters are 1-D, so excluded."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return [
        {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
    ]


def warmup_linear_decay(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_frac: float = 0.1
) -> LambdaLR:
    """Linear warmup from 0 over ``warmup_frac`` of the steps, then linear decay down to 0."""
    warmup = max(1, math.ceil(warmup_frac * total_steps))

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return LambdaLR(optimizer, factor)


def train_one(
    cfg: DictConfig,
    data: TokenData,
    make_encoder: Callable[[], BertModel],
    *,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[BertClassifier, FitResult, float]:
    """Fine-tune from the pretrained weights; also returns the mean seconds per epoch."""
    seed_everything(seed)
    recipe = cfg.recipe
    model = BertClassifier(make_encoder(), len(data.classes), float(recipe.dropout)).to(device)
    batch_size, max_epochs = int(recipe.batch_size), int(recipe.max_epochs)
    steps = math.ceil(len(data.train) / batch_size) * max_epochs
    optimizer = torch.optim.AdamW(param_groups(model, float(recipe.weight_decay)), lr=lr)
    scheduler = warmup_linear_decay(optimizer, steps, float(recipe.warmup_frac))
    train_cfg = TrainConfig(
        lr=lr,
        weight_decay=float(recipe.weight_decay),
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=int(recipe.patience),
    )
    start = time.perf_counter()
    result = fit(
        model,
        data.train,
        data.y_train,
        data.val,
        data.y_val,
        train_cfg,
        device,
        seed,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return model, result, (time.perf_counter() - start) / result.epochs_run


def run(
    cfg: DictConfig,
    data: TokenData,
    make_encoder: Callable[[], BertModel],
    device: torch.device,
    revision: str | None = None,
) -> dict[str, Any]:
    outputs = Path(to_absolute_path(cfg.outputs_dir))
    method, condition = str(cfg.method), str(cfg.condition)
    n_classes = len(data.classes)

    trials: list[dict[str, Any]] = []
    if cfg.lr is None:
        for lr in cfg.search.lr:
            _, result, seconds = train_one(
                cfg, data, make_encoder, lr=float(lr), seed=int(cfg.search.seed), device=device
            )
            trials.append(
                {
                    "lr": float(lr),
                    "val_top1": result.val_top1,
                    "val_loss": result.val_loss,
                    "best_epoch": result.best_epoch,
                    "seconds_per_epoch": seconds,
                }
            )
            log.info("lr=%g -> val top-1 %.4f (epoch %d)", lr, result.val_top1, result.best_epoch)
        chosen_lr = float(best_config(trials)["lr"])
    else:  # search skipped: `just train-bert-finetune lr=0.00003`
        chosen_lr = float(cfg.lr)
    log.info("chosen lr=%g", chosen_lr)

    finals: list[dict[str, Any]] = []
    for seed in (int(s) for s in cfg.seeds):
        model, result, seconds = train_one(
            cfg, data, make_encoder, lr=chosen_lr, seed=seed, device=device
        )
        write_preds(
            preds_path(outputs, method, seed, "val", condition),
            data.ids_val,
            predict(model, data.val, device),
            n_classes=n_classes,
        )
        write_preds(
            preds_path(outputs, method, seed, "test", condition),
            data.ids_test,
            predict(model, data.test, device),
            n_classes=n_classes,
        )
        save_classifier(ckpt_path(outputs, method, seed).with_suffix(""), model, lr=chosen_lr)
        finals.append(
            {
                "seed": seed,
                "val_top1": result.val_top1,
                "val_loss": result.val_loss,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "seconds_per_epoch": seconds,
            }
        )
        log.info("seed %d -> val top-1 %.4f, %.0f s/epoch", seed, result.val_top1, seconds)

    top1 = np.array([f["val_top1"] for f in finals])
    seconds_mean = float(np.mean([f["seconds_per_epoch"] for f in finals]))
    meta = {
        "encoder": str(cfg.model_name),
        "weights_revision": revision,
        "L": data.max_len,
        "preprocessing": PREPROCESSING,
        "pooling": "masked mean of last_hidden_state, [CLS]/[SEP] included",
        "recipe": dict(cfg.recipe),
        "chosen_lr": chosen_lr,
        "trials": trials,
        "finals": finals,
        "seconds_per_epoch_mean": seconds_mean,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_commit": git_commit(),
    }
    meta_path = outputs / "preds" / method / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {
        "trials": trials,
        "chosen_lr": chosen_lr,
        "finals": finals,
        "val_top1_mean": float(top1.mean()),
        "val_top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
        "seconds_per_epoch_mean": seconds_mean,
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/bert_finetune")
def main(cfg: DictConfig) -> None:
    device = pick_device(str(cfg.device))
    splits = load_splits(
        Path(to_absolute_path(cfg.paths.upmc_dir)), Path(to_absolute_path(cfg.val_ids))
    )
    if cfg.max_len is None:
        max_len = cache_max_len(Path(to_absolute_path(cfg.text_features)))
    else:
        max_len = int(cfg.max_len)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data = load_token_data(splits, tokenizer, max_len)

    def make_encoder() -> BertModel:
        return BertModel.from_pretrained(cfg.model_name, add_pooling_layer=False)

    summary = run(cfg, data, make_encoder, device, resolve_revision(cfg.model_name))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info(
        "val top-1 %.4f ± %.4f, %.0f s/epoch",
        summary["val_top1_mean"],
        summary["val_top1_std"],
        summary["seconds_per_epoch_mean"],
    )


if __name__ == "__main__":
    main()
