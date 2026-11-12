"""Unimodal references: MLP head on one frozen pooled feature cache.

The same code serves both modalities, so they differ only by their features:

- ``text_mlp`` (sheet B2 b): BERT pooled vectors, ``just train-text-mlp``;
- ``image_mlp`` (sheet C2): ViT CLS vectors, ``just train-image-mlp``.

Protocol 2.4: grid over (lr, weight decay) with seed 0, keep the best validation top-1, then
retrain that configuration with seeds 0, 1 and 2. Validation predictions and a checkpoint are
written for every seed; test predictions only for this final configuration. No test metric is
computed here.
"""

import itertools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import Features, labels_for, read_features
from recipe_fusion.models.heads import MLPHead, save_head
from recipe_fusion.preds import ckpt_path, preds_path, write_preds
from recipe_fusion.seed import pick_device, seed_everything
from recipe_fusion.training import ArrayBatches, FitResult, TrainConfig, fit, predict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PooledData:
    classes: list[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    ids_val: list[str]
    x_test: np.ndarray
    ids_test: list[str]


def read_aligned(features_dir: Path, split: str, samples: Sequence[Sample]) -> Features:
    """Read the cache of ``split`` and check it follows the order of the title file."""
    features = read_features(features_dir, split)
    if features.ids != [s.image_name for s in samples]:
        raise ValueError(f"{split}: cached ids do not follow {split}_titles.csv; re-extract")
    return features


def rows(features: Features, ids: Sequence[str]) -> np.ndarray:
    """Pooled vectors of ``ids``, in the order requested (alignment is by id)."""
    position = {image_id: i for i, image_id in enumerate(features.ids)}
    return np.asarray(features.pooled[[position[image_id] for image_id in ids]])


def load_pooled_data(splits: Splits, features_dir: Path) -> PooledData:
    f_train = read_aligned(features_dir, "train", splits.train_all)
    f_test = read_aligned(features_dir, "test", splits.test)
    ids_train = [s.image_name for s in splits.train]
    ids_val = [s.image_name for s in splits.val]
    return PooledData(
        classes=splits.classes,
        x_train=rows(f_train, ids_train),
        y_train=labels_for(ids_train, splits.train_all, splits.classes),
        x_val=rows(f_train, ids_val),
        y_val=labels_for(ids_val, splits.train_all, splits.classes),
        ids_val=ids_val,
        x_test=np.asarray(f_test.pooled),
        ids_test=f_test.ids,
    )


def best_config(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest validation top-1; ties go to the lower validation loss."""
    return max(trials, key=lambda t: (t["val_top1"], -t["val_loss"]))


def train_one(
    cfg: DictConfig,
    data: PooledData,
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> tuple[MLPHead, FitResult]:
    seed_everything(seed)
    model = MLPHead(
        dim_in=data.x_train.shape[1],
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
        ArrayBatches(data.x_train),
        data.y_train,
        ArrayBatches(data.x_val),
        data.y_val,
        train_cfg,
        device,
        seed,
    )
    return model, result


def run(cfg: DictConfig, data: PooledData, device: torch.device) -> dict[str, Any]:
    outputs = Path(to_absolute_path(cfg.outputs_dir))
    method, condition = str(cfg.method), str(cfg.condition)
    n_classes = len(data.classes)

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
    chosen = best_config(trials)
    log.info("chosen: lr=%g wd=%g", chosen["lr"], chosen["weight_decay"])

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
        val_logits = predict(model, ArrayBatches(data.x_val), device)
        test_logits = predict(model, ArrayBatches(data.x_test), device)
        write_preds(
            preds_path(outputs, method, seed, "val", condition),
            data.ids_val,
            val_logits,
            n_classes=n_classes,
        )
        write_preds(
            preds_path(outputs, method, seed, "test", condition),
            data.ids_test,
            test_logits,
            n_classes=n_classes,
        )
        save_head(
            ckpt_path(outputs, method, seed),
            model,
            lr=chosen["lr"],
            weight_decay=chosen["weight_decay"],
            seed=seed,
        )
        finals.append({"seed": seed, "val_top1": result.val_top1, "val_loss": result.val_loss})
        log.info("seed %d -> val top-1 %.4f", seed, result.val_top1)

    top1 = np.array([f["val_top1"] for f in finals])
    return {
        "trials": trials,
        "chosen": chosen,
        "finals": finals,
        "val_top1_mean": float(top1.mean()),
        "val_top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
    }


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/text_mlp")
def main(cfg: DictConfig) -> None:
    splits = load_splits(
        Path(to_absolute_path(cfg.paths.upmc_dir)), Path(to_absolute_path(cfg.val_ids))
    )
    data = load_pooled_data(splits, Path(to_absolute_path(cfg.features)))
    summary = run(cfg, data, pick_device(str(cfg.device)))
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("val top-1 %.4f ± %.4f", summary["val_top1_mean"], summary["val_top1_std"])


if __name__ == "__main__":
    main()
