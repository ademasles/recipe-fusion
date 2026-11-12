"""Frozen BERT text feature extraction (sheet B1).

Writes the feature cache of contract 2.2 to ``data/features/text/<encoder>/``.
"""

from __future__ import annotations

import html
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from recipe_fusion.data.upmc import read_split
from recipe_fusion.features.cache import write_features
from recipe_fusion.seed import pick_device

SPLITS = ("train", "test")  # train first, then test; file order, no shuffling

_RESIDUAL = re.compile(r"&#?\w+;|\\x[0-9a-fA-F]{2}")


def preprocess(title: str) -> str:
    """Only allowed preprocessing: the text stays as the model will see it."""
    return html.unescape(title).strip()


def round_up(x: float, multiple: int = 8) -> int:
    """Smallest multiple of ``multiple`` that is >= x (at least one multiple)."""
    return max(multiple, multiple * math.ceil(x / multiple))


def token_lengths(texts: list[str], tokenizer) -> np.ndarray:
    """WordPiece lengths, [CLS] and [SEP] included, without truncation."""
    ids = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    return np.array([len(x) for x in ids])


def truncated_pct(lengths: np.ndarray, max_len: int) -> float:
    return float(100.0 * np.mean(lengths > max_len))


def residual_escape_pct(texts: list[str]) -> float:
    """% of titles that still contain an HTML entity or a literal \\xNN."""
    return float(100.0 * np.mean([bool(_RESIDUAL.search(t)) for t in texts]))


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of (B, L, H) over the non-masked tokens -> (B, H)."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


@torch.inference_mode()
def encode(texts, tokenizer, model, max_len: int, batch_size: int = 256):
    """Return (pooled (N, H) f16, tokens (N, L, H) f16, mask (N, L) bool)."""
    model.eval()
    device = next(model.parameters()).device
    n, hidden = len(texts), model.config.hidden_size
    pooled = np.zeros((n, hidden), dtype=np.float16)
    tokens = np.zeros((n, max_len, hidden), dtype=np.float16)
    mask = np.zeros((n, max_len), dtype=np.bool_)

    for i in tqdm(range(0, n, batch_size), desc="encode", unit="batch"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        last = model(**enc).last_hidden_state.float()  # not pooler_output
        am = enc["attention_mask"]
        j = i + len(batch)
        pooled[i:j] = masked_mean(last, am).cpu().numpy().astype(np.float16)
        tokens[i:j] = last.cpu().numpy().astype(np.float16)
        mask[i:j] = am.bool().cpu().numpy()
    return pooled, tokens, mask


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def load_model(model_name: str, device):
    """Load tokenizer and frozen model (the only part that may download)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model


def resolve_revision(model_name: str, cache_dir: Path | None = None) -> str | None:
    """Hub revision of the cached weights (no network).

    Returns None for a local checkpoint or a model that is not in the HF cache.
    """
    if Path(model_name).is_dir():  # local checkpoint: no hub revision
        return None
    path = try_to_load_from_cache(model_name, "config.json", cache_dir=cache_dir)
    return Path(path).parent.name if isinstance(path, str) else None


def run(
    root: Path,
    out_dir: Path,
    cfg: Mapping[str, Any],
    tokenizer,
    model,
    device_name: str = "cpu",
    revision: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Full computation, without loading any model (testable on CPU).

    Returns the metadata written for each split.
    """

    samples = {s: list(read_split(root, s)) for s in SPLITS}
    texts = {s: [preprocess(x.text) for x in samples[s]] for s in SPLITS}
    lengths = {s: token_lengths(texts[s], tokenizer) for s in SPLITS}

    pct = float(cfg["length_percentile"])
    p_val = float(np.percentile(lengths["train"], pct))
    max_len = int(cfg.get("max_len") or round_up(p_val, int(cfg["length_multiple"])))
    batch_size = int(cfg["batch_size"])

    common: dict[str, Any] = {
        "encoder": cfg["model_name"],
        "weights_revision": revision,
        "d": int(model.config.hidden_size),
        "L": max_len,
        "length_percentile": pct,
        "train_length_at_percentile": p_val,
        "preprocessing": "html.unescape(title).strip()",
        "pooling": "masked mean of last_hidden_state, [CLS]/[SEP] included",
        "batch_size": batch_size,
        "dtype": "float16",
        "git_commit": git_commit(),
        "device": device_name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }

    metas: dict[str, dict[str, Any]] = {}
    for s in SPLITS:
        t0 = time.perf_counter()
        pooled, tokens, mask = encode(texts[s], tokenizer, model, max_len, batch_size)
        metas[s] = {
            **common,
            "n": len(samples[s]),
            "truncated_pct": truncated_pct(lengths[s], max_len),
            "residual_escape_pct": residual_escape_pct(texts[s]),
            "extraction_seconds": round(time.perf_counter() - t0, 2),
        }
        ids = [x.image_name for x in samples[s]]
        write_features(out_dir, s, ids, pooled, tokens, mask, meta=metas[s])
    return metas


@hydra.main(version_base=None, config_path="../../../configs", config_name="features/text")
def main(cfg: DictConfig) -> None:
    device = pick_device()
    print(f"device={device}", flush=True)
    tokenizer, model = load_model(cfg.model_name, device)
    revision = resolve_revision(cfg.model_name)
    root = Path(to_absolute_path(cfg.root))
    out_dir = Path(to_absolute_path(cfg.out_dir))
    meta = run(root, out_dir, cfg, tokenizer, model, str(device), revision)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
