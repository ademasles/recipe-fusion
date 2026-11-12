"""Prediction and checkpoint files of contract 2.3."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

N_CLASSES = 101


def preds_path(outputs_dir: Path, method: str, seed: int, split: str, condition: str) -> Path:
    return outputs_dir / "preds" / method / f"seed{seed}" / f"{split}_{condition}.npz"


def ckpt_path(outputs_dir: Path, method: str, seed: int) -> Path:
    return outputs_dir / "ckpt" / method / f"seed{seed}.pt"


def write_preds(
    path: Path,
    ids: Sequence[str],
    logits: npt.NDArray[np.float32],
    *,
    n_classes: int = N_CLASSES,
) -> None:
    """Validate, then write ``ids`` and float32 ``logits`` (N, n_classes), before softmax."""
    if len(set(ids)) != len(ids):
        raise ValueError("ids: duplicate image names")
    if logits.dtype != np.float32:
        raise TypeError(f"logits: expected float32, got {logits.dtype}")
    if logits.shape != (len(ids), n_classes):
        raise ValueError(f"logits: expected shape ({len(ids)}, {n_classes}), got {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("logits: contains NaN or infinite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), logits=logits)
