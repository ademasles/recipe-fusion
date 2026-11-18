import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from recipe_fusion import demo as D
from recipe_fusion.data.upmc import Sample
from recipe_fusion.models.cross_attention import CrossAttentionFusion, ModelConfig, TrainMeans
from recipe_fusion.models.heads import EarlyConcatHead, MLPHead
from recipe_fusion.models.late_fusion import Fusion

DIM, TEXT_LEN, IMAGE_LEN = 8, 4, 2
CLASSES = ["apple_pie", "pizza", "sushi"]
SMALL = ModelConfig(d_model=8, heads=2, layers=1, dropout=0.0, head_hidden=8, head_dropout=0.0)


def encoded(seed: int = 0, title: str | None = "apple pie") -> D.Encoded:
    rng = np.random.default_rng(seed)
    return D.Encoded(
        title=title,
        text_pooled=rng.normal(size=DIM).astype(np.float16),
        text_tokens=rng.normal(size=(TEXT_LEN, DIM)).astype(np.float16),
        text_mask=np.array([True, True, False, False]),
        image_pooled=rng.normal(size=DIM).astype(np.float16),
        image_grid=rng.normal(size=(IMAGE_LEN, DIM)).astype(np.float16),
    )


def means() -> D.TrainingMeans:
    return D.TrainingMeans(
        np.zeros(DIM, np.float16),
        np.ones(DIM, np.float16),
        TrainMeans(
            np.zeros((TEXT_LEN, DIM), np.float16),
            np.array([True, False, False, False]),
            np.ones((IMAGE_LEN, DIM), np.float16),
        ),
    )


def make_demo() -> D.Demo:
    torch.manual_seed(0)
    n = len(CLASSES)
    heads = {
        "image_mlp": MLPHead(DIM, 8, n, 0.0).eval(),
        "text_mlp": MLPHead(DIM, 8, n, 0.0).eval(),
        "early_concat": EarlyConcatHead(DIM, DIM, 8, n, 0.0).eval(),
        "cross_attention": CrossAttentionFusion(TEXT_LEN, IMAGE_LEN, DIM, n, SMALL).eval(),
    }
    return D.Demo(
        classes=CLASSES,
        device=torch.device("cpu"),
        max_len=TEXT_LEN,
        encode_title=lambda title: (np.zeros(DIM), np.zeros((TEXT_LEN, DIM)), np.ones(TEXT_LEN)),
        encode_image=lambda image: (np.zeros(DIM), np.zeros((IMAGE_LEN, DIM))),
        heads=heads,
        bert_ft=None,
        bert_tokenizer=None,
        late={"late_calibrated": (Fusion(2.0, 0.5, 0.3), "text_mlp")},
        means=means(),
    )


def test_predict_scores_every_model_and_fuses_the_unimodal_logits() -> None:
    demo = make_demo()
    logits = demo.predict(encoded())

    assert list(logits) == [
        "image_mlp",
        "text_mlp",
        "early_concat",
        "cross_attention",
        "late_calibrated",
    ]
    assert all(values.shape == (len(CLASSES),) for values in logits.values())
    expected = Fusion(2.0, 0.5, 0.3)(logits["image_mlp"][None], logits["text_mlp"][None])[0]
    np.testing.assert_allclose(logits["late_calibrated"], expected, rtol=1e-6)


def test_remove_replaces_one_modality_by_its_training_mean() -> None:
    pair, train = encoded(), means()

    no_text = D.remove(pair, "text", train)
    assert no_text.title is None
    np.testing.assert_array_equal(no_text.text_tokens, train.tokens.text)
    np.testing.assert_array_equal(no_text.text_mask, train.tokens.text_mask)
    np.testing.assert_array_equal(no_text.image_grid, pair.image_grid)

    no_image = D.remove(pair, "image", train)
    assert no_image.title == pair.title
    np.testing.assert_array_equal(no_image.image_pooled, train.image_pooled)
    np.testing.assert_array_equal(no_image.text_pooled, pair.text_pooled)

    with pytest.raises(ValueError, match="modality"):
        D.remove(pair, "audio", train)


def test_predict_changes_when_a_modality_is_removed() -> None:
    demo = make_demo()
    both = demo.predict(encoded())
    no_image = demo.predict(D.remove(encoded(), "image", demo.means))

    np.testing.assert_array_equal(no_image["text_mlp"], both["text_mlp"])
    assert not np.allclose(no_image["image_mlp"], both["image_mlp"])
    assert not np.allclose(no_image["cross_attention"], both["cross_attention"])


def test_training_means_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "demo" / "means.npz"
    means().save(path)
    loaded = D.TrainingMeans.load(path)

    np.testing.assert_array_equal(loaded.image_pooled, means().image_pooled)
    np.testing.assert_array_equal(loaded.tokens.text_mask, means().tokens.text_mask)


def test_load_cross_attention_restores_the_saved_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = CrossAttentionFusion(TEXT_LEN, IMAGE_LEN, DIM, len(CLASSES), SMALL).eval()
    path = tmp_path / "seed0.pt"
    torch.save({"state_dict": model.state_dict(), "model": asdict(SMALL), "val_top1": 0.5}, path)

    loaded = D.load_cross_attention(path, torch.device("cpu"))
    pair = encoded()
    inputs = {
        "text": torch.from_numpy(pair.text_tokens[None]).float(),
        "text_mask": torch.from_numpy(pair.text_mask[None]),
        "image": torch.from_numpy(pair.image_grid[None]).float(),
    }
    with torch.no_grad():
        torch.testing.assert_close(loaded(**inputs), model(**inputs))


def test_load_late_reads_the_fusion_fitted_for_the_seed(tmp_path: Path) -> None:
    meta = {
        "image": {"method": "image_mlp", "condition": "image_only"},
        "text": {"method": "text_bert_ft", "condition": "text_only"},
        "seeds": {"1": {"image_temperature": 1.5, "text_temperature": 0.8, "image_weight": 0.4}},
    }
    path = tmp_path / "preds" / "late_calibrated_ft" / "meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(meta), encoding="utf-8")

    assert D.load_late(tmp_path, "late_calibrated_ft", seed=1) == (
        Fusion(1.5, 0.8, 0.4),
        "text_bert_ft",
    )
    assert D.load_late(tmp_path, "late_calibrated", seed=1) is None


def test_pick_by_position_name_class_or_at_random() -> None:
    samples = [
        Sample(f"{label}_{i}.jpg", f"{label} {i}", label, "test")
        for label in CLASSES
        for i in range(3)
    ]

    assert D.pick(samples, 4) is samples[4]
    assert D.pick(samples, "sushi_1.jpg").image_name == "sushi_1.jpg"
    assert D.pick(samples, "pizza", seed=0).label == "pizza"
    assert D.pick(samples, None, seed=3) == D.pick(samples, None, seed=3)
    with pytest.raises(KeyError):
        D.pick(samples, "ramen")


def test_to_html_marks_right_and_wrong_top1() -> None:
    logits = {
        "image_mlp": np.array([3.0, 0.0, 0.0]),
        "text_mlp": np.array([0.0, 3.0, 0.0]),
    }
    page = D.to_html(logits, CLASSES, truth="apple_pie")

    assert page.count("<tr>") == 2
    assert page.index("✅") < page.index("❌")
    assert "Image only" in page


# --- End to end on a tiny dataset, with the two encoders replaced by small models ---------

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "apple", "pie", "pizza", "sushi", "easy"]
TITLES = {"apple_pie": "easy apple pie", "pizza": "pizza", "sushi": "easy sushi"}


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 768)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))[:, None, :].expand(-1, 577, -1)


class TinyVit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)[:, 0, :]


def to_tensor(image: D.Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255


def tiny_bert():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=768,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    return tokenizer, BertModel(config, add_pooling_layer=False).eval()


def test_load_demo_runs_live_and_agrees_with_the_test_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recipe_fusion.data.upmc import read_split
    from recipe_fusion.features import extract_image
    from recipe_fusion.features import text as T
    from recipe_fusion.features.cache import write_features
    from recipe_fusion.models.heads import save_head
    from recipe_fusion.preds import ckpt_path, preds_path, write_preds

    torch.manual_seed(0)
    upmc = tmp_path / "data" / "upmc-food101"
    names: dict[str, list[str]] = {}
    for split, per_class in (("train", 3), ("test", 1)):
        lines = []
        for c, label in enumerate(CLASSES):
            for i in range(per_class):
                name = f"{label}_{split}{i}.jpg"
                path = upmc / "images" / split / label / name
                path.parent.mkdir(parents=True, exist_ok=True)
                D.Image.new("RGB", (8, 8), color=(80 * c, 20 * i, 40)).save(path)
                lines.append(f"{name},{TITLES[label]},{label}")
        (upmc / "texts").mkdir(parents=True, exist_ok=True)
        (upmc / "texts" / f"{split}_titles.csv").write_text("\n".join(lines) + "\n")
        names[split] = [line.split(",")[0] for line in lines]
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "val_ids.txt").write_text("pizza_train2.jpg\n")
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path / "data"))

    tokenizer, bert = tiny_bert()
    vit = TinyVit().eval()
    monkeypatch.setattr(D.text_features, "load_model", lambda name, device: (tokenizer, bert))
    monkeypatch.setattr(D.extract_image, "load_encoder", lambda: (vit, to_tensor))

    features = tmp_path / "data" / "features"
    for split in ("train", "test"):
        samples = read_split(upmc, split)
        texts = [T.preprocess(s.text) for s in samples]
        pooled, tokens, mask = T.encode(texts, tokenizer, bert, max_len=TEXT_LEN)
        meta = {"L": TEXT_LEN, "encoder": "tiny-bert"}
        write_features(features / D.TEXT_DIR, split, names[split], pooled, tokens, mask, meta)
        grids: list[np.ndarray] = []
        images = [to_tensor(D.Image.open(s.image_path(upmc)).convert("RGB")) for s in samples]
        image_pooled = extract_image.encode(
            [torch.stack(images)],
            vit,
            token_callback=lambda seq, out=grids: out.append(extract_image.to_grid(seq)),
        )
        image_mask = np.ones((len(samples), 64), dtype=np.bool_)
        write_features(
            features / D.IMAGE_DIR, split, names[split], image_pooled, grids[0], image_mask
        )

    outputs = tmp_path / "outputs"
    n = len(CLASSES)
    save_head(ckpt_path(outputs, "image_mlp", 0), MLPHead(768, 8, n, 0.0))
    save_head(ckpt_path(outputs, "text_mlp", 0), MLPHead(768, 8, n, 0.0))
    cross = CrossAttentionFusion(TEXT_LEN, 64, 768, n, SMALL)
    path = outputs / "ckpt" / "cross_attention" / "seed0.pt"
    path.parent.mkdir(parents=True)
    torch.save({"state_dict": cross.state_dict(), "model": asdict(SMALL)}, path)

    demo = D.load_demo(tmp_path, seed=0, device="cpu")
    assert set(demo.heads) == {"image_mlp", "text_mlp", "cross_attention"}
    assert (outputs / "demo" / "train_means_seed0.npz").is_file()

    sample = D.pick(read_split(upmc, "test"), "sushi")
    pair = demo.encode(D.Image.open(sample.image_path(upmc)), sample.text)
    logits = demo.predict(pair)
    assert list(logits) == ["image_mlp", "text_mlp", "cross_attention"]

    test_logits = np.zeros((n, n), dtype=np.float32)
    test_logits[names["test"].index(sample.image_name)] = logits["image_mlp"]
    write_preds(
        preds_path(outputs, "image_mlp", 0, "test", "image_only"),
        names["test"],
        test_logits,
        n_classes=n,
    )
    report = {row["what"]: row for row in D.check_against_test(tmp_path, sample, pair, logits)}
    assert report["text features"]["max_abs_diff"] == 0
    assert report["image features"]["max_abs_diff"] == 0
    assert report["image_mlp"]["same_top1"] is True
    assert "text_mlp" not in report
