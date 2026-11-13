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


class EarlyConcatHead(nn.Module):
    """Early fusion head : one LayerNorm per modality, concatenation, then an MLP."""

    def __init__(
        self,
        dim_text: int = 768,
        dim_image: int = 768,
        hidden: int = 512,
        n_classes: int = 101,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.args = {
            "dim_text": dim_text,
            "dim_image": dim_image,
            "hidden": hidden,
            "n_classes": n_classes,
            "dropout": dropout,
        }
        self.norm_text = nn.LayerNorm(dim_text)
        self.norm_image = nn.LayerNorm(dim_image)
        self.mlp = nn.Sequential(
            nn.Linear(dim_text + dim_image, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([self.norm_text(text), self.norm_image(image)], dim=1))


def save_head(path: Path, model: MLPHead | EarlyConcatHead, **extra: float | int | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"args": model.args, "state_dict": model.state_dict(), "extra": extra}, path)


def load_head(path: Path, device: str | torch.device = "cpu") -> MLPHead:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MLPHead(**checkpoint["args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def load_early_concat(path: Path, device: str | torch.device = "cpu") -> EarlyConcatHead:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = EarlyConcatHead(**checkpoint["args"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
