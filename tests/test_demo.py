import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from recipe_fusion import demo as D
from recipe_fusion.features.text import preprocess
from recipe_fusion.models.bert_finetune import BertClassifier, save_classifier, tokenize
from recipe_fusion.models.heads import MLPHead, save_head
from recipe_fusion.models.late_fusion import Fusion
from recipe_fusion.preds import ckpt_path

CLASSES = ["apple_pie", "pizza", "sushi"]
WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "apple", "pie", "pizza", "sushi", "easy"]


class TinyVit(torch.nn.Module):
    """Stands in for ViT-B/16: an ``encoder`` and a (N, 768) output."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 768)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images.mean(dim=(2, 3)))


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Data, checkpoints and fusion meta of seed 0, with small models in place of ViT and BERT."""
    torch.manual_seed(0)
    lines = [f"{c}_{i}.jpg,{c} {i},{c}" for c in CLASSES for i in range(2)]
    texts = tmp_path / "data/upmc-food101/texts/train_titles.csv"
    texts.parent.mkdir(parents=True)
    texts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.delenv("F101_DATA_DIR", raising=False)

    outputs = tmp_path / "outputs"
    save_head(ckpt_path(outputs, "image_mlp", 0), MLPHead(768, 8, len(CLASSES), 0.0))
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    bert = BertClassifier(BertModel(config, add_pooling_layer=False), len(CLASSES), 0.0)
    save_classifier(ckpt_path(outputs, "text_bert_ft", 0).with_suffix(""), bert)
    for name, meta in (
        ("text_bert_ft", {"encoder": "tiny-bert", "L": 6}),
        (
            D.FUSION,
            {
                "seeds": {
                    "0": {"image_temperature": 2.0, "text_temperature": 0.5, "image_weight": 0.3}
                }
            },
        ),
    ):
        path = outputs / "preds" / name / "meta.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(D.extract_image, "load_encoder", lambda: (TinyVit(), to_tensor))
    monkeypatch.setattr(D.AutoTokenizer, "from_pretrained", lambda name: tiny_tokenizer())
    return tmp_path


def test_load_best_reads_the_seed_checkpoints_and_fitted_fusion(root: Path) -> None:
    model = D.load_best(root, seed=0, device="cpu")

    assert model.classes == CLASSES
    assert model.max_len == 6
    assert model.fusion == Fusion(2.0, 0.5, 0.3)


def test_predict_is_the_calibrated_fusion_of_both_branches(root: Path) -> None:
    model = D.load_best(root, seed=0, device="cpu")
    image, title = Image.new("RGB", (8, 8), color=(200, 120, 40)), "easy apple pie"

    probs = model.predict(image, title)

    with torch.no_grad():
        pooled = model.vit(to_tensor(image)[None]).half().float()
        image_logits = model.image_head(pooled).numpy()
        ids, mask = tokenize([preprocess(title)], model.tokenizer, model.max_len)
        text_logits = model.bert(torch.from_numpy(ids).long(), torch.from_numpy(mask).long())
    expected = D.softmax(model.fusion(image_logits, text_logits.numpy())[0])
    np.testing.assert_allclose(probs, expected, rtol=1e-5, atol=1e-6)
    assert probs.sum() == pytest.approx(1.0)


def test_top_k_sorts_by_probability() -> None:
    probs = np.array([0.2, 0.7, 0.1])

    assert D.top_k(probs, CLASSES, k=2) == [("pizza", 0.7), ("apple_pie", 0.2)]
