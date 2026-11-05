"""On-disk cache of frozen encoder features, shared by every model and fusion.

One directory per encoder, ``<features_dir>/<modality>/<encoder>/``, holding for each split::

    <split>_ids.txt      # one image name per line, in the order of texts/<split>_titles.csv
    <split>_pooled.npy   # float16 (N, d)
    <split>_tokens.npy   # float16 (N, L, d), optional
    <split>_mask.npy     # bool (N, L), True for real tokens; present iff tokens are
    meta.json            # {split: metadata}: encoder, preprocessing, extraction time...

Labels are not cached: they are read back from the title files with :func:`labels_for`.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from recipe_fusion.data.upmc import Sample

META_FILE = "meta.json"


@dataclass(frozen=True)
class Features:
    ids: list[str]
    pooled: npt.NDArray[np.float16]
    tokens: npt.NDArray[np.float16] | None = None
    mask: npt.NDArray[np.bool_] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)


def _paths(directory: Path, split: str) -> dict[str, Path]:
    return {
        name: directory / f"{split}_{name}.{ext}"
        for name, ext in (("ids", "txt"), ("pooled", "npy"), ("tokens", "npy"), ("mask", "npy"))
    }


def _check_dtype(name: str, array: np.ndarray, dtype: type[np.generic]) -> None:
    if array.dtype != dtype:
        raise TypeError(f"{name}: expected {np.dtype(dtype)}, got {array.dtype}")


def _check_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains NaN or infinite values")


def validate(features: Features) -> None:
    """Raise if ``features`` does not follow the cache format.

    Cheap structural checks run first, so a wrong dtype or shape fails fast before the
    full scans for non-finite values.
    """
    n = len(features.ids)
    if len(set(features.ids)) != n:
        raise ValueError("ids: duplicate image names")

    pooled, tokens, mask = features.pooled, features.tokens, features.mask
    _check_dtype("pooled", pooled, np.float16)
    if pooled.ndim != 2 or pooled.shape[0] != n:
        raise ValueError(f"pooled: expected shape ({n}, d), got {pooled.shape}")

    if (tokens is None) != (mask is None):
        raise ValueError("tokens and mask must be given together")
    if tokens is not None and mask is not None:
        _check_dtype("tokens", tokens, np.float16)
        _check_dtype("mask", mask, np.bool_)
        if tokens.ndim != 3 or tokens.shape[0] != n:
            raise ValueError(f"tokens: expected shape ({n}, L, d), got {tokens.shape}")
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"mask: expected shape {tokens.shape[:2]}, got {mask.shape}")
        if not mask.any(axis=1).all():
            raise ValueError("mask: some rows have no real token")

    _check_finite("pooled", pooled)
    if tokens is not None:
        _check_finite("tokens", tokens)


def write_features(
    directory: Path,
    split: str,
    ids: Sequence[str],
    pooled: npt.NDArray[np.float16],
    tokens: npt.NDArray[np.float16] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Features:
    """Validate the features of one split, then write them under ``directory``."""
    features = Features(list(ids), pooled, tokens, mask, dict(meta or {}))
    validate(features)

    directory.mkdir(parents=True, exist_ok=True)
    paths = _paths(directory, split)
    np.save(paths["pooled"], pooled)
    if tokens is not None and mask is not None:
        np.save(paths["tokens"], tokens)
        np.save(paths["mask"], mask)
    else:
        # A stale token file from an earlier run would otherwise be read back with these ids.
        paths["tokens"].unlink(missing_ok=True)
        paths["mask"].unlink(missing_ok=True)

    meta_path = directory / META_FILE
    all_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    all_meta[split] = features.meta
    meta_path.write_text(json.dumps(all_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Ids last: their presence marks the split as complete.
    paths["ids"].write_text("\n".join(features.ids) + "\n", encoding="utf-8")
    return features


def read_features(directory: Path, split: str, *, mmap: bool = True) -> Features:
    """Read the features of one split; arrays are memory-mapped unless ``mmap=False``."""
    paths = _paths(directory, split)
    if not paths["ids"].is_file():
        raise FileNotFoundError(f"no {split} features in {directory}")
    mode = "r" if mmap else None
    ids = paths["ids"].read_text(encoding="utf-8").splitlines()
    pooled = np.load(paths["pooled"], mmap_mode=mode)
    tokens = np.load(paths["tokens"], mmap_mode=mode) if paths["tokens"].is_file() else None
    mask = np.load(paths["mask"], mmap_mode=mode) if paths["mask"].is_file() else None
    meta_path = directory / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    return Features(ids, pooled, tokens, mask, meta.get(split, {}))


def labels_for(ids: Sequence[str], samples: Sequence[Sample], classes: Sequence[str]) -> np.ndarray:
    """Integer labels of ``ids``, using the position of each class name in ``classes``."""
    label_of = {sample.image_name: sample.label for sample in samples}
    index_of = {name: i for i, name in enumerate(classes)}
    missing = [image_id for image_id in ids if image_id not in label_of]
    if missing:
        raise KeyError(f"{len(missing)} ids not in the samples, e.g. {missing[0]!r}")
    return np.array([index_of[label_of[image_id]] for image_id in ids], dtype=np.int64)
