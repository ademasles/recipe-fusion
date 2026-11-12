"""MLP head and training protocol 2.4, shared by the text, image and fusion heads."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPHead(nn.Module):
    """LayerNorm(d) -> Linear(d, h) -> GELU -> Dropout(p) -> Linear(h, classes)."""

    def __init__(
        self, dim_in: int = 768, hidden: int = 512, n_classes: int = 101, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.args = {
            "dim_in": dim_in,
            "hidden": hidden,
            "n_classes": n_classes,
            "dropout": dropout,
        }
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_head(path: Path, model: MLPHead, **extra: float | int | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": model.args, "state_dict": model.state_dict(), "extra": extra}, path)


def load_head(path: Path, device: str | torch.device = "cpu") -> MLPHead:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLPHead(**checkpoint["args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


@torch.inference_mode()
def predict_logits(
    model: nn.Module, x: np.ndarray, *, device: torch.device, batch_size: int = 1024
) -> np.ndarray:
    """Float32 logits of ``model`` (eval mode) for the rows of ``x``."""
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(np.asarray(x[start : start + batch_size]))
        chunks.append(model(batch.to(device=device, dtype=torch.float32)).float().cpu())
    return torch.cat(chunks).numpy()


@dataclass(frozen=True)
class FitResult:
    best_epoch: int  # 1-based
    epochs_run: int
    val_loss: float
    val_top1: float


def fit(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
) -> FitResult:
    """AdamW, early stopping on the validation loss, best epoch kept (protocol 2.4).

    Call ``seed_everything(seed)`` before building ``model``. The model is left at the
    weights of its best validation-loss epoch.
    """
    model.to(device)
    x = torch.as_tensor(np.asarray(x_train)).to(device=device, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.long).to(device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)

    best: FitResult | None = None
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[batch]), y[batch]).backward()
            optimizer.step()

        logits = torch.from_numpy(predict_logits(model, x_val, device=device))
        val_loss = float(F.cross_entropy(logits, y_val_t))
        if best is None or val_loss < best.val_loss:
            val_top1 = float((logits.argmax(dim=1) == y_val_t).float().mean())
            best = FitResult(epoch, epoch, val_loss, val_top1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best is None:
        raise ValueError("max_epochs must be >= 1")
    model.load_state_dict(best_state)
    return replace(best, epochs_run=epoch)
