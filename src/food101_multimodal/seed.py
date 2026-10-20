"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device).

    With ``deterministic=True`` cuDNN autotuning is disabled and PyTorch is asked to use
    deterministic algorithms; operations without one only warn. Expect slower training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global state is what third-party code reads
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(preference: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for ``"auto"``."""
    if preference != "auto":
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
