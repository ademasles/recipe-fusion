"""Bidirectional cross-attention fusion of title tokens and the 8 x 8 image grid (A4).

Each modality is normalised and projected to ``d_model``, then, in every layer, title
tokens attend to image tokens and image tokens attend to title tokens (both directions
read the previous layer, so neither side goes first). The two sides are mean-pooled
(title padding excluded), concatenated and classified by the same head as the other
fusions. Run with ``just train-cross-attention``; see ``configs/train/cross_attention.yaml``.
"""

import csv
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor, nn

from recipe_fusion.data.split import read_ids
from recipe_fusion.data.upmc import class_names, read_split
from recipe_fusion.features.cache import Features, labels_for, read_features
from recipe_fusion.features.text import git_commit
from recipe_fusion.preds import write_preds
from recipe_fusion.seed import pick_device, seed_everything
from recipe_fusion.training import TrainConfig, fit, predict

log = logging.getLogger(__name__)

METHOD = "cross_attention"
CONDITIONS = ("both", "text_mean", "image_mean", "text_shuffled", "image_shuffled")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    heads: int = 4
    layers: int = 1
    dropout: float = 0.1
    head_hidden: int = 512
    head_dropout: float = 0.3


class CrossBlock(nn.Module):
    """Pre-norm cross-attention of ``query`` over ``context``, then a feed-forward layer."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, context: Tensor, context_padding: Tensor | None) -> Tensor:
        context = self.norm_context(context)
        attended, _ = self.attention(
            self.norm_query(query), context, context, key_padding_mask=context_padding
        )
        query = query + attended
        return query + self.ff(self.norm_ff(query))


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        text_len: int,
        image_len: int,
        feature_dim: int,
        n_classes: int,
        cfg: ModelConfig = ModelConfig(),  # noqa: B008 - frozen dataclass
    ) -> None:
        super().__init__()
        d = cfg.d_model
        self.text_in = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, d))
        self.image_in = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, d))
        self.text_position = nn.Parameter(torch.zeros(text_len, d))
        self.image_position = nn.Parameter(torch.zeros(image_len, d))
        nn.init.trunc_normal_(self.text_position, std=0.02)
        nn.init.trunc_normal_(self.image_position, std=0.02)
        self.text_to_image = nn.ModuleList(
            CrossBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)
        )
        self.image_to_text = nn.ModuleList(
            CrossBlock(d, cfg.heads, cfg.dropout) for _ in range(cfg.layers)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, n_classes),
        )

    def forward(self, text: Tensor, text_mask: Tensor, image: Tensor) -> Tensor:
        """``text`` (B, Lt, D) with ``text_mask`` (B, Lt) True on real tokens, ``image``
        (B, Li, D); returns logits (B, n_classes)."""
        text = self.text_in(text) + self.text_position
        image = self.image_in(image) + self.image_position
        padding = ~text_mask
        for to_image, to_text in zip(self.text_to_image, self.image_to_text, strict=True):
            text, image = to_image(text, image, None), to_text(image, text, padding)
        weights = text_mask.unsqueeze(-1).to(text.dtype)
        text_pooled = (text * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        return self.head(torch.cat([text_pooled, image.mean(1)], dim=1))


class FusionBatches:
    """Rows ``rows`` of the text and image caches of one split, read lazily from disk.

    ``text_override``/``image_override`` replace a modality for the ablations: either a
    single mean example broadcast to every row, or a permutation of the rows.
    """

    def __init__(self, text: Features, image: Features, rows: npt.NDArray[np.int64]) -> None:
        if text.tokens is None or text.mask is None or image.tokens is None:
            raise ValueError("cross-attention needs the token caches of both modalities")
        if text.ids != image.ids:
            raise ValueError("text and image caches list different ids")
        self.text_tokens, self.text_mask, self.image_tokens = text.tokens, text.mask, image.tokens
        self.rows = rows
        self.text_mean: tuple[np.ndarray, np.ndarray] | None = None
        self.image_mean: np.ndarray | None = None
        self.text_permutation: np.ndarray | None = None
        self.image_permutation: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def ablated(self, condition: str, means: "TrainMeans", seed: int = 0) -> "FusionBatches":
        view = FusionBatches.__new__(FusionBatches)
        view.__dict__.update(self.__dict__)
        permutation = np.random.default_rng(seed).permutation(len(self.rows))
        match condition:
            case "both":
                pass
            case "text_mean":
                view.text_mean = (means.text, means.text_mask)
            case "image_mean":
                view.image_mean = means.image
            case "text_shuffled":
                view.text_permutation = permutation
            case "image_shuffled":
                view.image_permutation = permutation
            case _:
                raise ValueError(f"unknown condition {condition!r}, expected one of {CONDITIONS}")
        return view

    def _source_rows(self, positions: np.ndarray, permutation: np.ndarray | None) -> np.ndarray:
        return self.rows[positions if permutation is None else permutation[positions]]

    @staticmethod
    def _read(array: np.ndarray, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows, kind="stable")
        out = np.empty((len(rows), *array.shape[1:]), dtype=array.dtype)
        out[order] = array[rows[order]]  # sorted reads from the memory map
        return out

    def get(self, rows: npt.NDArray[np.int64]) -> dict[str, Tensor]:
        n = len(rows)
        if self.text_mean is not None:
            tokens, mask = self.text_mean
            text = np.broadcast_to(tokens, (n, *tokens.shape))
            text_mask = np.broadcast_to(mask, (n, *mask.shape))
        else:
            source = self._source_rows(rows, self.text_permutation)
            text, text_mask = (
                self._read(self.text_tokens, source),
                self._read(self.text_mask, source),
            )
        if self.image_mean is not None:
            image = np.broadcast_to(self.image_mean, (n, *self.image_mean.shape))
        else:
            image = self._read(self.image_tokens, self._source_rows(rows, self.image_permutation))
        return {
            "text": torch.from_numpy(np.ascontiguousarray(text)),
            "text_mask": torch.from_numpy(np.ascontiguousarray(text_mask)),
            "image": torch.from_numpy(np.ascontiguousarray(image)),
        }


@dataclass(frozen=True)
class TrainMeans:
    """Mean title and mean image over the training rows, for the ``*_mean`` ablations."""

    text: np.ndarray  # (Lt, D), mean over the rows where the position is a real token
    text_mask: np.ndarray  # (Lt,), positions that are real tokens in at least half the rows
    image: np.ndarray  # (Li, D)

    @classmethod
    def compute(cls, data: FusionBatches, chunk: int = 1024) -> "TrainMeans":
        text_sum = np.zeros(data.text_tokens.shape[1:], dtype=np.float64)
        counts = np.zeros(data.text_mask.shape[1], dtype=np.float64)
        image_sum = np.zeros(data.image_tokens.shape[1:], dtype=np.float64)
        for start in range(0, len(data.rows), chunk):
            rows = data.rows[start : start + chunk]
            mask = np.asarray(data.text_mask[rows])
            text_sum += (
                np.asarray(data.text_tokens[rows], dtype=np.float32) * mask[..., None]
            ).sum(0)
            counts += mask.sum(0)
            image_sum += np.asarray(data.image_tokens[rows], dtype=np.float32).sum(0)
        text_mask = counts >= len(data.rows) / 2
        text_mask[0] = True  # [CLS] is always there; keeps at least one real token
        text = text_sum / np.maximum(counts, 1)[:, None]
        return cls(
            text.astype(np.float16),
            text_mask,
            (image_sum / len(data.rows)).astype(np.float16),
        )


@dataclass(frozen=True)
class Splits:
    train: FusionBatches
    val: FusionBatches
    test: FusionBatches
    train_labels: np.ndarray
    val_labels: np.ndarray
    val_ids: list[str]
    test_ids: list[str]
    n_classes: int


def load_splits(cfg: DictConfig) -> Splits:
    root = Path(to_absolute_path(cfg.paths.upmc_dir))
    text_dir = Path(to_absolute_path(cfg.text_features))
    image_dir = Path(to_absolute_path(cfg.image_features))
    train_samples, test_samples = read_split(root, "train"), read_split(root, "test")
    classes = class_names(train_samples)

    text, image = read_features(text_dir, "train"), read_features(image_dir, "train")
    val_set = set(read_ids(Path(to_absolute_path(cfg.val_ids))))
    is_val = np.array([image_id in val_set for image_id in text.ids])
    if is_val.sum() != len(val_set):
        raise ValueError("some validation ids are missing from the feature cache")
    labels = labels_for(text.ids, train_samples, classes)
    train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    test_text, test_image = read_features(text_dir, "test"), read_features(image_dir, "test")
    if set(test_text.ids) != {sample.image_name for sample in test_samples}:
        raise ValueError("the test feature cache does not match the test titles")
    test_rows = np.arange(len(test_text.ids))
    return Splits(
        train=FusionBatches(text, image, train_rows),
        val=FusionBatches(text, image, val_rows),
        test=FusionBatches(test_text, test_image, test_rows),
        train_labels=labels[train_rows],
        val_labels=labels[val_rows],
        val_ids=[text.ids[i] for i in val_rows],
        test_ids=list(test_text.ids),
        n_classes=len(classes),
    )


def build_model(splits: Splits, cfg: ModelConfig) -> CrossAttentionFusion:
    return CrossAttentionFusion(
        text_len=splits.train.text_tokens.shape[1],
        image_len=splits.train.image_tokens.shape[1],
        feature_dim=splits.train.text_tokens.shape[2],
        n_classes=splits.n_classes,
        cfg=cfg,
    )


def train_one(
    splits: Splits, model_cfg: ModelConfig, train_cfg: TrainConfig, seed: int, device: torch.device
) -> tuple[CrossAttentionFusion, dict[str, float]]:
    seed_everything(seed)
    model = build_model(splits, model_cfg)
    result = fit(
        model,
        splits.train,
        splits.train_labels,
        splits.val,
        splits.val_labels,
        train_cfg,
        device,
        seed,
    )
    summary = {
        "lr": train_cfg.lr,
        "weight_decay": train_cfg.weight_decay,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "val_loss": result.val_loss,
        "val_top1": result.val_top1,
    }
    return model, summary


def search(
    splits: Splits, cfg: DictConfig, model_cfg: ModelConfig, device: torch.device, out: Path
) -> TrainConfig:
    """Protocol 2.4: the 6 (lr, weight decay) pairs with seed 0, best by validation top-1."""
    rows = []
    for lr, wd in itertools.product(cfg.search.lr, cfg.search.weight_decay):
        train_cfg = TrainConfig(lr=float(lr), weight_decay=float(wd), **cfg.protocol)
        _, summary = train_one(splits, model_cfg, train_cfg, seed=0, device=device)
        log.info("search lr=%g wd=%g: val top1 %.4f", lr, wd, summary["val_top1"])
        rows.append(summary)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "search.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["val_top1"])
    log.info(
        "best: lr=%g wd=%g (val top1 %.4f)", best["lr"], best["weight_decay"], best["val_top1"]
    )
    return TrainConfig(lr=best["lr"], weight_decay=best["weight_decay"], **cfg.protocol)


def final(
    splits: Splits,
    cfg: DictConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> None:
    """The chosen configuration with each seed: checkpoint, validation and test predictions."""
    preds = Path(to_absolute_path(cfg.preds_dir)) / METHOD
    ckpt = Path(to_absolute_path(cfg.ckpt_dir)) / METHOD
    ckpt.mkdir(parents=True, exist_ok=True)
    means = TrainMeans.compute(splits.train)
    for seed in cfg.seeds:
        model, summary = train_one(splits, model_cfg, train_cfg, int(seed), device)
        torch.save(
            {"state_dict": model.state_dict(), "model": asdict(model_cfg), **summary},
            ckpt / f"seed{seed}.pt",
        )
        write_preds(
            preds / f"seed{seed}" / "val_both.npz",
            splits.val_ids,
            predict(model, splits.val, device),
            n_classes=splits.n_classes,
        )
        for condition in CONDITIONS:
            logits = predict(model, splits.test.ablated(condition, means), device)
            write_preds(
                preds / f"seed{seed}" / f"test_{condition}.npz",
                splits.test_ids,
                logits,
                n_classes=splits.n_classes,
            )
        log.info(
            "seed %d: val top1 %.4f at epoch %d", seed, summary["val_top1"], summary["best_epoch"]
        )
    (preds / "meta.json").write_text(
        json.dumps(
            {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "seeds": list(cfg.seeds),
                "git_commit": git_commit(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run(cfg: DictConfig) -> None:
    device = pick_device(str(cfg.device))
    model_cfg = ModelConfig(**cfg.model)
    splits = load_splits(cfg)
    log.info("train %d, val %d, test %d", len(splits.train), len(splits.val), len(splits.test))
    if cfg.lr is None or cfg.weight_decay is None:
        train_cfg = search(splits, cfg, model_cfg, device, Path(to_absolute_path(cfg.search_dir)))
    else:
        train_cfg = TrainConfig(
            lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), **cfg.protocol
        )
    final(splits, cfg, model_cfg, train_cfg, device)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/cross_attention")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
