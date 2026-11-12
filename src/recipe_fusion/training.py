"""Training protocol 2.4, shared by every head trained on the feature cache.

AdamW, batch 256, at most 50 epochs, early stopping on the validation loss (patience 5),
keeping the best validation point. A model is any ``nn.Module`` whose ``forward`` takes
the tensors of a batch as keyword arguments and returns logits.
"""

import copy
import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

log = logging.getLogger(__name__)


class Batches(Protocol):
    """Rows of a feature cache, fetched by position."""

    def __len__(self) -> int: ...

    def get(self, rows: npt.NDArray[np.int64]) -> dict[str, Tensor]: ...


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    weight_decay: float
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5


@dataclass
class FitResult:
    best_epoch: int
    val_loss: float
    val_top1: float
    history: list[dict[str, float]] = field(default_factory=list)


def _to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    non_blocking = device.type == "cuda"
    return {
        name: (value.float() if value.is_floating_point() else value).to(
            device, non_blocking=non_blocking
        )
        for name, value in batch.items()
    }


def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def iterate(
    n: int, batch_size: int, rng: np.random.Generator | None = None
) -> Iterator[np.ndarray]:
    """Row positions by batch, shuffled when ``rng`` is given; sorted inside each batch so
    that reads from a memory-mapped cache stay sequential."""
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield np.sort(order[start : start + batch_size])


@torch.no_grad()
def predict(
    model: nn.Module, data: Batches, device: torch.device, batch_size: int = 1024
) -> npt.NDArray[np.float32]:
    """Logits (N, n_classes) in float32, in the row order of ``data``."""
    model.eval()
    outputs = []
    for rows in iterate(len(data), batch_size):
        with _autocast(device):
            logits = model(**_to_device(data.get(rows), device))
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _evaluate(
    model: nn.Module, data: Batches, labels: np.ndarray, device: torch.device
) -> tuple[float, float]:
    logits = torch.from_numpy(predict(model, data, device))
    target = torch.from_numpy(labels)
    loss = functional.cross_entropy(logits, target).item()
    top1 = (logits.argmax(dim=1) == target).float().mean().item()
    return loss, top1


def fit(
    model: nn.Module,
    train: Batches,
    train_labels: np.ndarray,
    val: Batches,
    val_labels: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> FitResult:
    """Train with protocol 2.4 and leave ``model`` at its best validation epoch."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    labels = torch.from_numpy(train_labels)
    best_loss, best_state, best = math.inf, copy.deepcopy(model.state_dict()), None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for rows in iterate(len(train), cfg.batch_size, rng):
            batch = _to_device(train.get(rows), device)
            target = labels[rows].to(device)
            with _autocast(device):
                loss = functional.cross_entropy(model(**batch), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
            count += len(rows)
        val_loss, val_top1 = _evaluate(model, val, val_labels, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "val_loss": val_loss,
                "val_top1": val_top1,
            }
        )
        log.info(
            "epoch %d: train loss %.4f, val loss %.4f, val top1 %.4f",
            epoch,
            total / count,
            val_loss,
            val_top1,
        )
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            best = FitResult(epoch, val_loss, val_top1)
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    assert best is not None
    best.history = history
    return best


def write_preds(path: Path, ids: list[str], logits: np.ndarray) -> None:
    """Predictions in format 2.3: ``ids`` and float32 ``logits`` (N, n_classes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), logits=logits.astype(np.float32))
