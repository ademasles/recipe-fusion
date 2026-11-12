"""Late fusion of two unimodal prediction files (methods ``late_calibrated``, ``late_sum``).

Nothing is trained. For each seed, on the validation split only:

1. one temperature per modality minimises the validation NLL (grid of log T in [-2, 2]);
2. the image weight ``w`` in {0, 0.05, ..., 1} maximises the validation top-1 of
   ``w * log_softmax(image / T_img) + (1 - w) * log_softmax(text / T_txt)``,
   ties broken by the lower NLL.

The fitted fusion is then frozen and applied to the test split, under ``both`` and under the
same test-time ablations as the cross-attention, so the two can be compared directly:

- ``*_shuffled``: a unimodal model scores each sample independently, so shuffling its input
  is exactly permuting its logit rows (same permutation as the cross-attention);
- ``*_mean``: the unimodal head applied to the mean training feature, one row broadcast to
  every sample. Needs the head checkpoint; skipped for methods without one (TF-IDF).

``late_sum`` adds the raw logits, without calibration, under the same conditions.
"""

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.evaluate import align, load_preds
from recipe_fusion.features.cache import labels_for, read_features
from recipe_fusion.models.heads import load_head
from recipe_fusion.preds import ckpt_path, preds_path, write_preds

log = logging.getLogger(__name__)

Combine = Callable[[np.ndarray, np.ndarray], npt.NDArray[np.float32]]

CONDITIONS = ("both", "text_mean", "image_mean", "text_shuffled", "image_shuffled")
LOG_TEMPERATURES = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.1), 10)
WEIGHTS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 10)


def log_softmax(logits: np.ndarray) -> npt.NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(-log_softmax(logits)[np.arange(len(labels)), labels].mean())


def calibrate_temperature(
    logits: np.ndarray, labels: np.ndarray, log_temperatures: np.ndarray = LOG_TEMPERATURES
) -> tuple[float, float]:
    """Temperature with the lowest validation NLL; ties go to the one closest to 1."""
    loss, _, log_t = min(
        (nll(logits / math.exp(lt), labels), abs(lt), lt) for lt in map(float, log_temperatures)
    )
    return math.exp(log_t), loss


@dataclass(frozen=True)
class Fusion:
    image_temperature: float
    text_temperature: float
    image_weight: float

    def __call__(self, image: np.ndarray, text: np.ndarray) -> npt.NDArray[np.float32]:
        fused = self.image_weight * log_softmax(image / self.image_temperature) + (
            1 - self.image_weight
        ) * log_softmax(text / self.text_temperature)
        return fused.astype(np.float32)


def choose_weight(
    image: np.ndarray,
    text: np.ndarray,
    labels: np.ndarray,
    image_temperature: float,
    text_temperature: float,
) -> tuple[float, float, float]:
    """Image weight with the best validation top-1, then the lowest NLL: (w, top-1, NLL)."""
    candidates = []
    for weight in map(float, WEIGHTS):
        fused = Fusion(image_temperature, text_temperature, weight)(image, text)
        top1 = float((fused.argmax(1) == labels).mean())
        candidates.append((-top1, nll(fused, labels), weight))
    neg_top1, loss, weight = min(candidates)
    return weight, -neg_top1, loss


def fit_fusion(image: np.ndarray, text: np.ndarray, labels: np.ndarray) -> tuple[Fusion, dict]:
    t_img, nll_img = calibrate_temperature(image, labels)
    t_txt, nll_txt = calibrate_temperature(text, labels)
    weight, top1, loss = choose_weight(image, text, labels, t_img, t_txt)
    fusion = Fusion(t_img, t_txt, weight)
    stats = {"image_nll": nll_img, "text_nll": nll_txt, "val_top1": top1, "val_nll": loss}
    return fusion, {**asdict(fusion), **stats}


def read_logits(
    outputs: Path, method: str, seed: int, split: str, condition: str, ids: list[str], n: int
) -> np.ndarray:
    path = preds_path(outputs, method, seed, split, condition)
    if not path.is_file():
        raise FileNotFoundError(f"missing predictions {path}; run {method} first")
    return align(*load_preds(path), ids, n)


def mean_row(
    outputs: Path, method: str, seed: int, features_dir: Path, train_ids: list[str]
) -> np.ndarray | None:
    """Logits of the unimodal head on the mean training feature, or None without a head."""
    checkpoint = ckpt_path(outputs, method, seed)
    if not checkpoint.is_file():
        return None
    features = read_features(features_dir, "train")
    position = {image_id: i for i, image_id in enumerate(features.ids)}
    rows = np.sort([position[image_id] for image_id in train_ids])
    total = np.zeros(features.pooled.shape[1], dtype=np.float64)
    for start in range(0, len(rows), 4096):
        total += np.asarray(features.pooled[rows[start : start + 4096]], dtype=np.float64).sum(0)
    mean = torch.from_numpy((total / len(rows)).astype(np.float32))[None]
    head = load_head(checkpoint).eval()
    with torch.no_grad():
        return head(mean).numpy()


def ablate(
    logits: np.ndarray, kind: str, permutation: np.ndarray, mean: np.ndarray | None
) -> np.ndarray | None:
    """Unimodal test logits under ``kind`` ("mean" or "shuffled"); None if not available."""
    if kind == "shuffled":
        return logits[permutation]
    if mean is None:
        return None
    return np.broadcast_to(mean, logits.shape)


def run(cfg: DictConfig, splits: Splits) -> dict[str, Any]:
    outputs = Path(to_absolute_path(cfg.outputs_dir))
    n = len(splits.classes)
    val_ids = [s.image_name for s in splits.val]
    test_ids = [s.image_name for s in splits.test]
    train_ids = [s.image_name for s in splits.train]
    y_val = labels_for(val_ids, splits.train_all, splits.classes)
    # Same permutation as the cross-attention test ablations (seed 0, test file order).
    permutation = np.random.default_rng(0).permutation(len(test_ids))

    meta: dict[str, Any] = {
        "image": {"method": str(cfg.image.method), "condition": str(cfg.image.condition)},
        "text": {"method": str(cfg.text.method), "condition": str(cfg.text.condition)},
        "seeds": {},
    }
    for seed in (int(s) for s in cfg.seeds):
        logits = {
            (modality, split): read_logits(
                outputs,
                str(cfg[modality].method),
                seed,
                split,
                str(cfg[modality].condition),
                ids,
                n,
            )
            for modality in ("image", "text")
            for split, ids in (("val", val_ids), ("test", test_ids))
        }
        fusion, stats = fit_fusion(logits["image", "val"], logits["text", "val"], y_val)
        meta["seeds"][str(seed)] = stats
        log.info("seed %d: %s", seed, json.dumps(stats))

        combine: dict[str, Combine] = {str(cfg.name): fusion}
        if cfg.sum_name:
            combine[str(cfg.sum_name)] = lambda image, text: (image + text).astype(np.float32)
        for name in combine:
            out = combine[name](logits["image", "val"], logits["text", "val"])
            write_preds(preds_path(outputs, name, seed, "val", "both"), val_ids, out, n_classes=n)

        means = {
            modality: mean_row(
                outputs,
                str(cfg[modality].method),
                seed,
                Path(to_absolute_path(cfg[modality].features)),
                train_ids,
            )
            for modality in ("image", "text")
        }
        for condition in CONDITIONS:
            image, text = logits["image", "test"], logits["text", "test"]
            if condition != "both":
                modality, kind = condition.split("_")
                ablated = ablate(logits[modality, "test"], kind, permutation, means[modality])
                if ablated is None:
                    log.info("seed %d: %s skipped, %s has no head", seed, condition, modality)
                    continue
                image, text = (ablated, text) if modality == "image" else (image, ablated)
            for name, combine_fn in combine.items():
                write_preds(
                    preds_path(outputs, name, seed, "test", condition),
                    test_ids,
                    combine_fn(image, text),
                    n_classes=n,
                )

    for name in [str(cfg.name)] + ([str(cfg.sum_name)] if cfg.sum_name else []):
        path = outputs / "preds" / name / "meta.json"
        entry = meta if name == cfg.name else {k: meta[k] for k in ("image", "text")}
        path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    top1 = np.array([s["val_top1"] for s in meta["seeds"].values()])
    spread = top1.std(ddof=1) if len(top1) > 1 else 0.0
    log.info("%s val top-1 %.4f ± %.4f", cfg.name, top1.mean(), spread)
    return meta


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/late_fusion")
def main(cfg: DictConfig) -> None:
    splits = load_splits(
        Path(to_absolute_path(cfg.paths.upmc_dir)), Path(to_absolute_path(cfg.val_ids))
    )
    run(cfg, splits)


if __name__ == "__main__":
    main()
