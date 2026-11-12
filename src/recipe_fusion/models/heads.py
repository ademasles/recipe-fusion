"""MLP head shared by the unimodal references and the early fusion (sheets B2, B3, C2)."""

from pathlib import Path

import torch
from torch import nn


class MLPHead(nn.Module):
    """LayerNorm(d) -> Linear(d, h) -> GELU -> Dropout(p) -> Linear(h, classes)."""

    def __init__(
        self, dim_in: int = 768, hidden: int = 512, n_classes: int = 101, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.args = {"dim_in": dim_in, "hidden": hidden, "n_classes": n_classes, "dropout": dropout}
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
