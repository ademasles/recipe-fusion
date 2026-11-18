"""Live prediction of the best model on one image and title pair (notebooks/live_demo.ipynb).

The best model is the calibrated late fusion of the fine-tuned BERT and the MLP on frozen
ViT-B/16 features (94.7 % test top-1). Both branches run on the raw image and title; the
temperatures and the image weight are the ones fitted on validation.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch import nn
from transformers import AutoTokenizer

from recipe_fusion.data.upmc import class_names, read_split
from recipe_fusion.features import extract_image
from recipe_fusion.features.text import preprocess
from recipe_fusion.models.bert_finetune import TokenBatches, load_classifier, tokenize
from recipe_fusion.models.heads import load_head
from recipe_fusion.models.late_fusion import Fusion
from recipe_fusion.preds import ckpt_path
from recipe_fusion.seed import pick_device
from recipe_fusion.training import ArrayBatches, predict

FUSION = "late_calibrated_ft"


def data_dir(root: Path) -> Path:
    """UPMC Food-101 directory, as in ``configs/paths``: ``F101_DATA_DIR`` or ``<root>/data``."""
    return Path(os.environ.get("F101_DATA_DIR", root / "data")) / "upmc-food101"


@dataclass
class BestModel:
    classes: list[str]
    device: torch.device
    vit: nn.Module
    transform: Any
    image_head: nn.Module
    bert: nn.Module
    tokenizer: Any
    max_len: int
    fusion: Fusion

    def predict(self, image: Image.Image, title: str) -> npt.NDArray[np.float64]:
        """Class probabilities (n_classes,) for one image and its title."""
        pixels = self.transform(image.convert("RGB"))[None]
        features = extract_image.encode([pixels], self.vit, device=self.device)
        image_logits = predict(self.image_head, ArrayBatches(features), self.device)
        ids, mask = tokenize([preprocess(title)], self.tokenizer, self.max_len)
        text_logits = predict(self.bert, TokenBatches(ids, mask), self.device)
        return softmax(self.fusion(image_logits, text_logits)[0])


def softmax(scores: np.ndarray) -> npt.NDArray[np.float64]:
    shifted = np.asarray(scores, dtype=np.float64) - np.max(scores)
    return np.exp(shifted) / np.exp(shifted).sum()


def top_k(probs: np.ndarray, classes: list[str], k: int = 5) -> list[tuple[str, float]]:
    return [(classes[int(i)], float(probs[i])) for i in np.argsort(-probs)[:k]]


def load_best(root: Path, seed: int = 0, device: str = "auto") -> BestModel:
    """Checkpoints of ``seed`` under ``<root>/outputs``, written by the training recipes."""
    outputs, torch_device = root / "outputs", pick_device(device)
    bert_meta = json.loads((outputs / "preds/text_bert_ft/meta.json").read_text(encoding="utf-8"))
    fusion_meta = json.loads((outputs / "preds" / FUSION / "meta.json").read_text(encoding="utf-8"))
    params = fusion_meta["seeds"][str(seed)]
    vit, transform = extract_image.load_encoder()
    return BestModel(
        classes=class_names(read_split(data_dir(root), "train")),
        device=torch_device,
        vit=vit.to(torch_device).eval(),
        transform=transform,
        image_head=load_head(ckpt_path(outputs, "image_mlp", seed), torch_device).eval(),
        bert=load_classifier(
            ckpt_path(outputs, "text_bert_ft", seed).with_suffix(""), torch_device
        ).eval(),
        tokenizer=AutoTokenizer.from_pretrained(bert_meta["encoder"]),
        max_len=int(bert_meta["L"]),
        fusion=Fusion(
            params["image_temperature"], params["text_temperature"], params["image_weight"]
        ),
    )
