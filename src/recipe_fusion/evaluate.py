"""Shared evaluation: every method is scored by this script and nothing else.

Predictions live in ``<preds_dir>/<method>/seed<k>/<split>_<condition>.npz`` with two arrays:
``ids`` (image names) and ``logits`` (float, N x n_classes, before softmax). Run with
``just evaluate`` (validation) or ``just evaluate split=test`` (final table only).

Outputs, in ``<results_dir>``:

- ``<split>.csv``: one row per method, seed, condition, subset and metric;
- ``<split>_summary.csv``: mean and sample standard deviation over seeds.
"""

import csv
import logging
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import numpy.typing as npt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.split import read_ids, split_train_val
from recipe_fusion.data.upmc import class_names, read_split

log = logging.getLogger(__name__)

PRED_FILE = re.compile(r"(?P<split>val|test)_(?P<condition>\w+)\.npz")
SEED_DIR = re.compile(r"seed(?P<seed>\d+)")


@dataclass(frozen=True)
class Run:
    method: str
    seed: int
    condition: str
    path: Path


def load_preds(path: Path) -> tuple[list[str], npt.NDArray[np.floating]]:
    with np.load(path, allow_pickle=False) as data:
        return [str(i) for i in data["ids"]], np.asarray(data["logits"])


def align(
    ids: Sequence[str], logits: np.ndarray, expected: Sequence[str], n_classes: int
) -> np.ndarray:
    """Logits reordered to ``expected``; raise unless ids match exactly and logits are sane."""
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in predictions")
    if set(ids) != set(expected):
        missing, extra = set(expected) - set(ids), set(ids) - set(expected)
        raise ValueError(f"ids do not match the split: {len(missing)} missing, {len(extra)} extra")
    if logits.shape != (len(ids), n_classes):
        raise ValueError(f"logits: expected shape ({len(ids)}, {n_classes}), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")
    row = {image_id: i for i, image_id in enumerate(ids)}
    return logits[[row[image_id] for image_id in expected]]


def metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Top-1, top-5 accuracy and macro F1 over all classes."""
    n_classes = logits.shape[1]
    top5 = np.argpartition(-logits, min(5, n_classes) - 1, axis=1)[:, :5]
    pred = logits.argmax(axis=1)
    return {
        "top1": float((pred == labels).mean()),
        "top5": float((top5 == labels[:, None]).any(axis=1).mean()),
        "macro_f1": macro_f1(pred, labels, n_classes),
    }


def macro_f1(pred: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1 over all classes; a class never seen nor predicted is 0."""
    tp = np.bincount(labels[pred == labels], minlength=n_classes)
    predicted = np.bincount(pred, minlength=n_classes)
    actual = np.bincount(labels, minlength=n_classes)
    denom = predicted + actual  # 2 TP + FP + FN
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test on paired per-example correctness of two methods."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return {"a_only": b, "b_only": c, "p_value": p}


def find_runs(preds_dir: Path, split: str) -> list[Run]:
    runs = []
    for path in sorted(preds_dir.glob(f"*/seed*/{split}_*.npz")):
        file_match = PRED_FILE.fullmatch(path.name)
        seed_match = SEED_DIR.fullmatch(path.parent.name)
        if file_match is None or seed_match is None:
            log.warning("skipping %s: name does not follow the predictions layout", path)
            continue
        seed = int(seed_match["seed"])
        runs.append(Run(path.parent.parent.name, seed, file_match["condition"], path))
    return runs


def evaluate(
    runs: Iterable[Run],
    ids: Sequence[str],
    labels: np.ndarray,
    subsets: Mapping[str, Iterable[str]],
    n_classes: int,
) -> list[dict[str, object]]:
    """Long-format rows: one per run, subset and metric. Subset ``all`` is always included."""
    position = {image_id: i for i, image_id in enumerate(ids)}
    masks = {"all": np.ones(len(ids), dtype=bool)}
    for name, members in subsets.items():
        mask = np.zeros(len(ids), dtype=bool)
        mask[[position[m] for m in members if m in position]] = True
        masks[name] = mask
    rows: list[dict[str, object]] = []
    for run in runs:
        run_ids, logits = load_preds(run.path)
        aligned = align(run_ids, logits, ids, n_classes)
        for subset, mask in masks.items():
            if not mask.any():
                continue
            for metric, value in metrics(aligned[mask], labels[mask]).items():
                rows.append(
                    {
                        "method": run.method,
                        "seed": run.seed,
                        "condition": run.condition,
                        "subset": subset,
                        "n": int(mask.sum()),
                        "metric": metric,
                        "value": value,
                    }
                )
    return rows


def aggregate(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Mean and sample standard deviation (ddof=1) over seeds."""
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["condition"], row["subset"], row["metric"])
        value = row["value"]
        if not isinstance(value, float):
            raise TypeError(f"metric value must be a float, got {value!r}")
        groups[key].append(value)
    return [
        {
            "method": method,
            "condition": condition,
            "subset": subset,
            "metric": metric,
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "seeds": len(values),
        }
        for (method, condition, subset, metric), values in sorted(groups.items(), key=str)
    ]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_ids_and_labels(cfg: DictConfig) -> tuple[list[str], np.ndarray, int]:
    root = Path(cfg.paths.upmc_dir)
    train = read_split(root, "train")
    classes = class_names(train)
    if cfg.split == "val":
        _, samples = split_train_val(train, read_ids(Path(to_absolute_path(cfg.val_ids))))
    elif cfg.split == "test":
        samples = read_split(root, "test")
    else:
        raise ValueError(f"split must be 'val' or 'test', got {cfg.split!r}")
    index = {name: i for i, name in enumerate(classes)}
    labels = np.array([index[s.label] for s in samples], dtype=np.int64)
    return [s.image_name for s in samples], labels, len(classes)


def run_evaluation(cfg: DictConfig) -> list[dict[str, object]]:
    ids, labels, n_classes = split_ids_and_labels(cfg)
    subsets = {
        name: read_ids(Path(to_absolute_path(path))) for name, path in (cfg.subsets or {}).items()
    }
    runs = find_runs(Path(to_absolute_path(cfg.preds_dir)), cfg.split)
    if not runs:
        raise FileNotFoundError(f"no {cfg.split} predictions under {cfg.preds_dir}")
    rows = evaluate(runs, ids, labels, subsets, n_classes)
    summary = aggregate(rows)
    results_dir = Path(to_absolute_path(cfg.results_dir))
    write_csv(results_dir / f"{cfg.split}.csv", rows)
    write_csv(results_dir / f"{cfg.split}_summary.csv", summary)
    for row in summary:
        if row["subset"] == "all" and row["metric"] == "top1":
            log.info(
                "%-24s %-16s top1 %.4f ± %.4f (%d seeds)",
                row["method"],
                row["condition"],
                row["mean"],
                row["std"],
                row["seeds"],
            )
    return summary


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
