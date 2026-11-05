"""Validation split drawn once from the official training set, stratified by class.

Run with ``just make-split``. The resulting ``splits/val_ids.txt`` is committed and never
redrawn: every model is selected on the same validation set, and the test split is only used
for the final table.
"""

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.upmc import Sample, read_split

log = logging.getLogger(__name__)


def make_val_split(samples: Sequence[Sample], frac: float = 0.1, seed: int = 0) -> list[str]:
    """Image names of a stratified validation subset, sorted.

    Each class keeps ``round(frac * n)`` of its ``n`` samples for validation, clamped so that
    both sides get at least one. The result depends only on the names, ``frac`` and ``seed``,
    not on the order of ``samples``.
    """
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    names = [sample.image_name for sample in samples]
    if len(set(names)) != len(names):
        raise ValueError("image names are not unique; they cannot serve as ids")

    by_class: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label].append(sample.image_name)

    rng = np.random.default_rng(seed)
    val: list[str] = []
    for label in sorted(by_class):
        class_names = sorted(by_class[label])
        if len(class_names) < 2:
            raise ValueError(f"class {label!r} has fewer than 2 samples")
        n_val = min(max(round(frac * len(class_names)), 1), len(class_names) - 1)
        picked = rng.permutation(len(class_names))[:n_val]
        val.extend(class_names[i] for i in picked)
    return sorted(val)


def split_train_val(
    samples: Iterable[Sample], val_ids: Collection[str]
) -> tuple[list[Sample], list[Sample]]:
    """Split ``samples`` into (train, validation), keeping their order."""
    val_set = set(val_ids)
    train, val = [], []
    for sample in samples:
        (val if sample.image_name in val_set else train).append(sample)
    if len(val) != len(val_set):
        raise ValueError(f"{len(val_set) - len(val)} validation ids are not in the samples")
    return train, val


def write_ids(path: Path, ids: Iterable[str], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; the validation split is drawn only once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image_id}\n" for image_id in ids), encoding="utf-8")


def read_ids(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def make_split(cfg: DictConfig) -> Path:
    samples = read_split(Path(cfg.paths.upmc_dir), "train")
    val_ids = make_val_split(samples, cfg.split.frac, cfg.split.seed)
    out = Path(to_absolute_path(cfg.split.val_ids))
    write_ids(out, val_ids)
    train, val = split_train_val(samples, val_ids)
    log.info(
        "train %d, val %d (frac %.2f, seed %d) -> %s",
        len(train),
        len(val),
        cfg.split.frac,
        cfg.split.seed,
        out,
    )
    return out


@hydra.main(version_base=None, config_path="../../../configs", config_name="split")
def main(cfg: DictConfig) -> None:
    make_split(cfg)


if __name__ == "__main__":
    main()
