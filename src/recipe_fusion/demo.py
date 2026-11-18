"""Live prediction on one image and title pair, for the defence demo.

Used by ``notebooks/live_demo.ipynb``. Both encoders run on the raw image and title, exactly as
in the feature extraction, and every trained model scores the result: no cached feature or
prediction file is read for the pair itself. Removing a modality replaces it by its training
mean, as in the test-time ablations.

Needs the data, the feature cache (for the training means and the title length) and the
checkpoints written by the training recipes under ``outputs/``.
"""

import html
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoTokenizer

from recipe_fusion.data.loading import load_splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features import extract_image
from recipe_fusion.features import text as text_features
from recipe_fusion.features.cache import read_features
from recipe_fusion.models.bert_finetune import BertClassifier, load_classifier, tokenize
from recipe_fusion.models.cross_attention import (
    CrossAttentionFusion,
    FusionBatches,
    ModelConfig,
    TrainMeans,
)
from recipe_fusion.models.heads import load_early_concat, load_head
from recipe_fusion.models.late_fusion import Fusion
from recipe_fusion.preds import ckpt_path, preds_path
from recipe_fusion.seed import pick_device
from recipe_fusion.training import predict

TEXT_DIR = Path("text/bert-base-uncased")
IMAGE_DIR = Path("image/vit_b_16_swag")

# Display order and labels; a method whose checkpoint is missing is skipped.
LABELS = {
    "image_mlp": "Image only",
    "text_mlp": "Text only (frozen BERT)",
    "text_bert_ft": "Text only (fine-tuned BERT)",
    "early_concat": "Early fusion",
    "cross_attention": "Cross-attention",
    "late_calibrated": "Late fusion (frozen BERT + image)",
    "late_calibrated_ft": "Late fusion (fine-tuned BERT + image)",
}
MODALITIES = ("text", "image")


@dataclass(frozen=True)
class Paths:
    upmc_dir: Path
    features_dir: Path
    outputs_dir: Path
    val_ids: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        """The defaults of ``configs/paths``: ``F101_DATA_DIR`` or ``<root>/data``."""
        data = Path(os.environ.get("F101_DATA_DIR", root / "data"))
        return cls(
            data / "upmc-food101", data / "features", root / "outputs", root / "splits/val_ids.txt"
        )


@dataclass(frozen=True)
class Encoded:
    """One pair through both frozen encoders, in the float16 layout of the feature cache.

    ``title`` is kept for the fine-tuned BERT, which reads the text itself; it is None when
    the text is replaced by its training mean.
    """

    title: str | None
    text_pooled: npt.NDArray[np.float16]  # (D,)
    text_tokens: npt.NDArray[np.float16]  # (L, D)
    text_mask: npt.NDArray[np.bool_]  # (L,)
    image_pooled: npt.NDArray[np.float16]  # (D,)
    image_grid: npt.NDArray[np.float16]  # (64, D)


@dataclass(frozen=True)
class TrainingMeans:
    """Training means used to remove a modality: pooled vectors and token sequences."""

    text_pooled: npt.NDArray[np.float16]
    image_pooled: npt.NDArray[np.float16]
    tokens: TrainMeans

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            text_pooled=self.text_pooled,
            image_pooled=self.image_pooled,
            text=self.tokens.text,
            text_mask=self.tokens.text_mask,
            image=self.tokens.image,
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingMeans":
        with np.load(path) as f:
            return cls(
                f["text_pooled"],
                f["image_pooled"],
                TrainMeans(f["text"], f["text_mask"], f["image"]),
            )


class DictBatches:
    """Rows of a few aligned arrays, served under their names (one pair at a time here)."""

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = dict(arrays)

    def __len__(self) -> int:
        return len(next(iter(self.arrays.values())))

    def get(self, rows: npt.NDArray[np.int64]) -> dict[str, Tensor]:
        return {
            name: torch.from_numpy(np.ascontiguousarray(array[rows]))
            for name, array in self.arrays.items()
        }


def remove(encoded: Encoded, modality: str, means: TrainingMeans) -> Encoded:
    """``encoded`` with ``modality`` ("text" or "image") replaced by its training mean."""
    if modality == "text":
        return replace(
            encoded,
            title=None,
            text_pooled=means.text_pooled,
            text_tokens=means.tokens.text,
            text_mask=means.tokens.text_mask,
        )
    if modality == "image":
        return replace(encoded, image_pooled=means.image_pooled, image_grid=means.tokens.image)
    raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")


@dataclass
class Demo:
    classes: list[str]
    device: torch.device
    max_len: int
    encode_title: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]]
    encode_image: Callable[[Image.Image], tuple[np.ndarray, np.ndarray]]
    heads: dict[str, nn.Module]  # image_mlp, text_mlp, early_concat, cross_attention
    bert_ft: BertClassifier | None
    bert_tokenizer: Any
    late: dict[str, tuple[Fusion, str]]  # name -> (fitted fusion, text method)
    means: TrainingMeans

    def encode(self, image: Image.Image, title: str) -> Encoded:
        text_pooled, text_tokens, text_mask = self.encode_title(title)
        image_pooled, image_grid = self.encode_image(image)
        return Encoded(title, text_pooled, text_tokens, text_mask, image_pooled, image_grid)

    def _logits(self, model: nn.Module, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        return predict(model, DictBatches(arrays), self.device)[0]

    def predict(self, encoded: Encoded) -> dict[str, np.ndarray]:
        """Logits (n_classes,) of every available model, in the order of ``LABELS``."""
        text, image = encoded.text_pooled[None], encoded.image_pooled[None]
        logits: dict[str, np.ndarray] = {}
        if "image_mlp" in self.heads:
            logits["image_mlp"] = self._logits(self.heads["image_mlp"], {"x": image})
        if "text_mlp" in self.heads:
            logits["text_mlp"] = self._logits(self.heads["text_mlp"], {"x": text})
        if self.bert_ft is not None and encoded.title is not None:
            ids, mask = tokenize(
                [text_features.preprocess(encoded.title)], self.bert_tokenizer, self.max_len
            )
            arrays = {"input_ids": ids.astype(np.int64), "attention_mask": mask.astype(np.int64)}
            logits["text_bert_ft"] = self._logits(self.bert_ft, arrays)
        if "early_concat" in self.heads:
            arrays = {"text": text, "image": image}
            logits["early_concat"] = self._logits(self.heads["early_concat"], arrays)
        if "cross_attention" in self.heads:
            arrays = {
                "text": encoded.text_tokens[None],
                "text_mask": encoded.text_mask[None],
                "image": encoded.image_grid[None],
            }
            logits["cross_attention"] = self._logits(self.heads["cross_attention"], arrays)
        for name, (fusion, text_method) in self.late.items():
            if "image_mlp" in logits and text_method in logits:
                logits[name] = fusion(logits["image_mlp"][None], logits[text_method][None])[0]
        return {name: logits[name] for name in LABELS if name in logits}


def load_cross_attention(path: Path, device: torch.device) -> CrossAttentionFusion:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint["state_dict"]
    text_len, feature_dim = state["text_position"].shape[0], state["text_in.0.weight"].shape[0]
    model = CrossAttentionFusion(
        text_len=text_len,
        image_len=state["image_position"].shape[0],
        feature_dim=feature_dim,
        n_classes=state["head.4.weight"].shape[0],
        cfg=ModelConfig(**checkpoint["model"]),
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def load_late(outputs_dir: Path, name: str, seed: int) -> tuple[Fusion, str] | None:
    """The fusion fitted on validation for ``seed``, and the text method it combines."""
    path = outputs_dir / "preds" / name / "meta.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    params = meta["seeds"][str(seed)]
    fusion = Fusion(params["image_temperature"], params["text_temperature"], params["image_weight"])
    return fusion, meta["text"]["method"]


def training_means(paths: Paths, cache: Path) -> TrainingMeans:
    """Means over the training rows (validation excluded), computed once then cached."""
    if cache.is_file():
        return TrainingMeans.load(cache)
    splits = load_splits(paths.upmc_dir, paths.val_ids)
    text = read_features(paths.features_dir / TEXT_DIR, "train")
    image = read_features(paths.features_dir / IMAGE_DIR, "train")
    position = {image_id: i for i, image_id in enumerate(text.ids)}
    rows = np.sort([position[s.image_name] for s in splits.train]).astype(np.int64)
    pooled = {
        name: np.asarray(
            sum(
                np.asarray(f.pooled[rows[i : i + 8192]], dtype=np.float64).sum(0)
                for i in range(0, len(rows), 8192)
            )
            / len(rows)
        ).astype(np.float16)
        for name, f in (("text", text), ("image", image))
    }
    means = TrainingMeans(
        pooled["text"], pooled["image"], TrainMeans.compute(FusionBatches(text, image, rows))
    )
    means.save(cache)
    return means


def load_demo(root: Path, seed: int = 0, device: str = "auto") -> Demo:
    """Encoders, every checkpoint of ``seed`` found under ``outputs/``, and the training means."""
    paths = Paths.from_root(root)
    torch_device = pick_device(device)
    text_meta = read_features(paths.features_dir / TEXT_DIR, "test").meta
    max_len, model_name = int(text_meta["L"]), str(text_meta["encoder"])

    tokenizer, bert = text_features.load_model(model_name, torch_device)

    def encode_title(title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pooled, tokens, mask = text_features.encode(
            [text_features.preprocess(title)], tokenizer, bert, max_len, batch_size=1
        )
        return pooled[0], tokens[0], mask[0]

    vit, transform = extract_image.load_encoder()
    vit = vit.to(torch_device).eval()

    def encode_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        grids: list[np.ndarray] = []
        pixels = transform(image.convert("RGB"))[None]
        pooled = extract_image.encode(
            [pixels],
            vit,
            device=torch_device,
            token_callback=lambda seq: grids.append(extract_image.to_grid(seq)),
        )
        return pooled[0], grids[0][0]

    outputs = paths.outputs_dir
    heads: dict[str, nn.Module] = {}
    for method in ("image_mlp", "text_mlp"):
        if (path := ckpt_path(outputs, method, seed)).is_file():
            heads[method] = load_head(path, torch_device).eval()
    if (path := ckpt_path(outputs, "early_concat", seed)).is_file():
        heads["early_concat"] = load_early_concat(path, torch_device).eval()
    if (path := outputs / "ckpt" / "cross_attention" / f"seed{seed}.pt").is_file():
        heads["cross_attention"] = load_cross_attention(path, torch_device)
    bert_dir = ckpt_path(outputs, "text_bert_ft", seed).with_suffix("")
    bert_ft = load_classifier(bert_dir, torch_device).eval() if bert_dir.is_dir() else None

    late = {
        name: fitted
        for name in ("late_calibrated", "late_calibrated_ft")
        if (fitted := load_late(outputs, name, seed)) is not None
    }
    return Demo(
        classes=load_splits(paths.upmc_dir, paths.val_ids).classes,
        device=torch_device,
        max_len=max_len,
        encode_title=encode_title,
        encode_image=encode_image,
        heads=heads,
        bert_ft=bert_ft,
        bert_tokenizer=AutoTokenizer.from_pretrained(model_name) if bert_ft else None,
        late=late,
        means=training_means(paths, outputs / "demo" / f"train_means_seed{seed}.npz"),
    )


def pick(samples: Sequence[Sample], choice: int | str | None, seed: int | None = None) -> Sample:
    """A test pair by position, by image file name, by class name (random pair of that class),
    or at random when ``choice`` is None."""
    rng = np.random.default_rng(seed)
    if choice is None:
        return samples[int(rng.integers(len(samples)))]
    if isinstance(choice, int):
        return samples[choice]
    by_name = [s for s in samples if s.image_name == choice]
    if by_name:
        return by_name[0]
    of_class = [s for s in samples if s.label == choice]
    if of_class:
        return of_class[int(rng.integers(len(of_class)))]
    raise KeyError(f"{choice!r} is neither an image file name nor a class name")


def softmax(logits: np.ndarray) -> npt.NDArray[np.float64]:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits)
    return np.exp(shifted) / np.exp(shifted).sum()


def to_html(
    logits: Mapping[str, np.ndarray], classes: Sequence[str], truth: str | None, k: int = 3
) -> str:
    """One row per model: its top-``k`` classes with probability bars, the correct one in green."""
    rows = []
    for name, values in logits.items():
        probs = softmax(values)
        cells = []
        for rank, index in enumerate(np.argsort(-probs)[:k]):
            label, p = classes[int(index)], float(probs[index])
            colour = "#15803d" if label == truth else ("#b91c1c" if rank == 0 else "#6b7280")
            weight = "600" if rank == 0 else "400"
            cells.append(
                f'<td style="padding:4px 10px;min-width:170px">'
                f'<div style="color:{colour};font-weight:{weight}">{html.escape(label)}'
                f" <span style='color:#6b7280;font-weight:400'>{100 * p:.0f}%</span></div>"
                f'<div style="background:{colour};height:6px;width:{max(2, 160 * p):.0f}px;'
                f'border-radius:3px"></div></td>'
            )
        top = classes[int(np.argmax(values))]
        mark = "" if truth is None else ("✅" if top == truth else "❌")
        rows.append(
            f'<tr><td style="padding:4px 10px;white-space:nowrap">'
            f"{html.escape(LABELS.get(name, name))}</td>"
            f'<td style="padding:4px 6px">{mark}</td>{"".join(cells)}</tr>'
        )
    return f'<table style="border-collapse:collapse;font-size:14px">{"".join(rows)}</table>'


CONDITION = {"image_mlp": "image_only", "text_mlp": "text_only", "text_bert_ft": "text_only"}


def check_against_test(
    root: Path, sample: Sample, encoded: Encoded, logits: Mapping[str, np.ndarray], seed: int = 0
) -> list[dict[str, object]]:
    """Compare the live pass on a test pair with the cached features and the saved test
    predictions of that pair: largest absolute difference, and whether the top-1 agrees."""
    paths = Paths.from_root(root)
    report: list[dict[str, object]] = []
    for name, directory, live in (
        ("text features", TEXT_DIR, encoded.text_pooled),
        ("image features", IMAGE_DIR, encoded.image_pooled),
    ):
        cached = read_features(paths.features_dir / directory, "test")
        row = cached.ids.index(sample.image_name)
        diff = np.abs(np.asarray(cached.pooled[row], np.float32) - live.astype(np.float32))
        report.append({"what": name, "max_abs_diff": float(diff.max()), "same_top1": None})
    for method, values in logits.items():
        path = preds_path(paths.outputs_dir, method, seed, "test", CONDITION.get(method, "both"))
        if not path.is_file():
            continue
        with np.load(path) as f:
            ids = list(f["ids"])
            saved = f["logits"][ids.index(sample.image_name)]
        report.append(
            {
                "what": method,
                "max_abs_diff": float(np.abs(saved - values).max()),
                "same_top1": bool(saved.argmax() == values.argmax()),
            }
        )
    return report
