"""Early fusion of the frozen BERT and ViT pooled vectors (sheet B3, method ``early_concat``).

Protocol 2.4, as for the unimodal references: 6 trials with seed 0, then the best configuration
with seeds 0, 1 and 2. Ablations: the model trained on both modalities is evaluated on val and
test with one modality replaced by its training mean or shuffled across examples. No test metric
is computed here.
"""

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor

from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.features.cache import labels_for
from recipe_fusion.features.text import git_commit
from recipe_fusion.models.heads import EarlyConcatHead, save_head
from recipe_fusion.models.pooled_mlp import best_config, read_aligned
from recipe_fusion.models.pooled_mlp import rows as select_rows
from recipe_fusion.preds import ckpt_path, preds_path, write_preds
from recipe_fusion.seed import pick_device, seed_everything
from recipe_fusion.training import FitResult, TrainConfig, fit, predict

log = logging.getLogger(__name__)

CONDITIONS = ("both", "text_mean", "image_mean", "text_shuffled", "image_shuffled")


class PairBatches:
    """Rows of two aligned arrays, served as ``{"text": tensor, "image": tensor}``."""

    def __init__(self, text: np.ndarray, image: np.ndarray) -> None:
        if len(text) != len(image):
            raise ValueError(f"text has {len(text)} rows, image has {len(image)}")
        self.text = text
        self.image = image

    def __len__(self) -> int:
        return len(self.text)

    def get(self, rows: npt.NDArray[np.int64]) -> dict[str, Tensor]:
        return {
            "text": torch.from_numpy(np.ascontiguousarray(self.text[rows])),
            "image": torch.from_numpy(np.ascontiguousarray(self.image[rows])),
        }


@dataclass(frozen=True)
class FusionData:
    classes: list[str]
    text_train: np.ndarray
    image_train: np.ndarray
    y_train: np.ndarray
    text_val: np.ndarray
    image_val: np.ndarray
    y_val: np.ndarray
    ids_val: list[str]
    text_test: np.ndarray
    image_test: np.ndarray
    ids_test: list[str]


def load_fusion_data(splits: Splits, text_dir: Path, image_dir: Path) -> FusionData:
    """Pooled vectors of both modalities, aligned to the samples by image name."""
    text_train = read_aligned(text_dir, "train", splits.train_all)
    image_train = read_aligned(image_dir, "train", splits.train_all)
    text_test = read_aligned(text_dir, "test", splits.test)
    image_test = read_aligned(image_dir, "test", splits.test)
    ids_train = [s.image_name for s in splits.train]
    ids_val = [s.image_name for s in splits.val]
    return FusionData(
        classes=splits.classes,
        text_train=select_rows(text_train, ids_train),
        image_train=select_rows(image_train, ids_train),
        y_train=labels_for(ids_train, splits.train_all, splits.classes),
        text_val=select_rows(text_train, ids_val),
        image_val=select_rows(image_train, ids_val),
        y_val=labels_for(ids_val, splits.train_all, splits.classes),
        ids_val=ids_val,
        text_test=np.asarray(text_test.pooled),
        image_test=np.asarray(image_test.pooled),
        ids_test=text_test.ids,
    )


def ablate(
    text: np.ndarray,
    image: np.ndarray,
    condition: str,
    *,
    mean_text: np.ndarray,
    mean_image: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Text and image arrays of one condition; the inputs are left untouched."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
    order = np.random.default_rng(seed).permutation(len(text))
    if condition == "text_mean":
        return np.broadcast_to(mean_text, text.shape), image
    if condition == "image_mean":
        return text, np.broadcast_to(mean_image, image.shape)
    if condition == "text_shuffled":
        return text[order], image
    if condition == "image_shuffled":
        return text, image[order]
    return text, image


def train_one(
    cfg: DictConfig,
    data: FusionData,
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[EarlyConcatHead, FitResult]:
    seed_everything(seed)
    model = EarlyConcatHead(
        dim_text=data.text_train.shape[1],
        dim_image=data.image_train.shape[1],
        hidden=int(cfg.head.hidden),
        n_classes=len(data.classes),
        dropout=float(cfg.head.dropout),
    )
    train_cfg = TrainConfig(
        lr=lr,
        weight_decay=weight_decay,
        batch_size=int(cfg.train.batch_size),
        max_epochs=int(cfg.train.max_epochs),
        patience=int(cfg.train.patience),
    )
    result = fit(
        model,
        PairBatches(data.text_train, data.image_train),
        data.y_train,
        PairBatches(data.text_val, data.image_val),
        data.y_val,
        train_cfg,
        device,
        seed,
    )
    return model, result


def sweep(
    cfg: DictConfig, data: FusionData, device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Protocol 2.4: the 6 (lr, weight decay) pairs with seed 0, best by validation top-1."""
    trials: list[dict[str, Any]] = []
    for lr, weight_decay in itertools.product(cfg.sweep.lr, cfg.sweep.weight_decay):
        _, result = train_one(
            cfg,
            data,
            lr=float(lr),
            weight_decay=float(weight_decay),
            seed=int(cfg.sweep.seed),
            device=device,
        )
        trials.append(
            {
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "val_top1": result.val_top1,
                "val_loss": result.val_loss,
                "best_epoch": result.best_epoch,
            }
        )
        log.info("lr=%g wd=%g -> val top-1 %.4f", lr, weight_decay, result.val_top1)
    return trials, best_config(trials)


def run(cfg: DictConfig, data: FusionData, device: torch.device) -> dict[str, Any]:
    outputs = Path(to_absolute_path(cfg.outputs_dir))
    method = str(cfg.method)
    n_classes = len(data.classes)

    if cfg.lr is None or cfg.weight_decay is None:
        trials, chosen = sweep(cfg, data, device)
    else:  # search skipped: `just train-early-fusion lr=0.0003 weight_decay=0.01`
        trials, chosen = [], {"lr": float(cfg.lr), "weight_decay": float(cfg.weight_decay)}
    log.info("chosen: lr=%g wd=%g", chosen["lr"], chosen["weight_decay"])

    mean_text = data.text_train.mean(axis=0, dtype=np.float32)
    mean_image = data.image_train.mean(axis=0, dtype=np.float32)
    evaluated = {
        "val": (data.text_val, data.image_val, data.ids_val),
        "test": (data.text_test, data.image_test, data.ids_test),
    }

    finals: list[dict[str, Any]] = []
    for seed in (int(s) for s in cfg.seeds):
        model, result = train_one(
            cfg,
            data,
            lr=chosen["lr"],
            weight_decay=chosen["weight_decay"],
            seed=seed,
            device=device,
        )
        val_top1: dict[str, float] = {}
        for split, (text, image, ids) in evaluated.items():
            for condition in CONDITIONS:
                text_c, image_c = ablate(
                    text,
                    image,
                    condition,
                    mean_text=mean_text,
                    mean_image=mean_image,
                    seed=int(cfg.ablation_seed),  # fixed, as in cross_attention
                )
                logits = predict(model, PairBatches(text_c, image_c), device)
                write_preds(
                    preds_path(outputs, method, seed, split, condition),
                    ids,
                    logits,
                    n_classes=n_classes,
                )
                if split == "val":
                    val_top1[condition] = float((logits.argmax(axis=1) == data.y_val).mean())
        save_head(
            ckpt_path(outputs, method, seed),
            model,
            lr=chosen["lr"],
            weight_decay=chosen["weight_decay"],
            seed=seed,
        )
        finals.append({"seed": seed, "val_loss": result.val_loss, "val_top1": val_top1})
        log.info("seed %d -> val top-1 (both) %.4f", seed, val_top1["both"])

    meta_path = outputs / "preds" / method / "meta.json"
    meta = {"chosen": chosen, "seeds": [f["seed"] for f in finals], "git_commit": git_commit()}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        values = np.array([f["val_top1"][condition] for f in finals])
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[condition] = {"mean": float(values.mean()), "std": std}
    return {"trials": trials, "chosen": chosen, "finals": finals, "val_top1": summary}


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/early_concat")
def main(cfg: DictConfig) -> None:
    splits = load_splits(
        Path(to_absolute_path(cfg.paths.upmc_dir)), Path(to_absolute_path(cfg.val_ids))
    )
    data = load_fusion_data(
        splits,
        Path(to_absolute_path(cfg.text_features)),
        Path(to_absolute_path(cfg.image_features)),
    )
    summary = run(cfg, data, pick_device(str(cfg.device)))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for condition, stats in summary["val_top1"].items():
        log.info("val top-1 %-14s %.4f ± %.4f", condition, stats["mean"], stats["std"])


if __name__ == "__main__":
    main()
