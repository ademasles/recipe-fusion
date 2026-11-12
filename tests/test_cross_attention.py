from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from recipe_fusion.evaluate import align, load_preds
from recipe_fusion.features.cache import write_features
from recipe_fusion.models.cross_attention import (
    CONDITIONS,
    CrossAttentionFusion,
    ModelConfig,
    TrainMeans,
    load_splits,
    run,
)
from recipe_fusion.training import TrainConfig, fit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TINY = ModelConfig(d_model=16, heads=2, layers=1, head_hidden=32)
LT, LI, D = 6, 4, 8


def tiny_model(n_classes: int = 3) -> CrossAttentionFusion:
    torch.manual_seed(0)
    return CrossAttentionFusion(LT, LI, D, n_classes, TINY).eval()


def test_forward_shape_and_padding_is_ignored() -> None:
    model = tiny_model()
    text, image = torch.randn(2, LT, D), torch.randn(2, LI, D)
    mask = torch.tensor([[True] * 3 + [False] * 3, [True] * LT])
    logits = model(text, mask, image)
    assert logits.shape == (2, 3)

    noisy = text.clone()
    noisy[0, 3:] = 1e3  # padded positions of the first title
    torch.testing.assert_close(model(noisy, mask, image), logits)


def test_both_directions_are_used() -> None:
    model = tiny_model()
    text, image = torch.randn(1, LT, D), torch.randn(1, LI, D)
    mask = torch.ones(1, LT, dtype=torch.bool)
    base = model(text, mask, image)
    assert not torch.allclose(model(text + torch.randn_like(text), mask, image), base)
    assert not torch.allclose(model(text, mask, image + torch.randn_like(image)), base)


def write_dataset(root: Path, n_per_class: int = 12) -> list[str]:
    """Separable toy caches: each class adds its own direction to the tokens."""
    rng = np.random.default_rng(0)
    text_centers, image_centers = 3 * rng.standard_normal((2, 3, 1, D))
    classes = ["apple_pie", "pho", "sushi"]
    titles = root / "upmc-food101" / "texts"
    titles.mkdir(parents=True)
    val_ids = []
    for split, n in (("train", n_per_class), ("test", 4)):
        names, labels = [], []
        for k, label in enumerate(classes):
            for i in range(n):
                names.append(f"{label}_{split}{i}.jpg")
                labels.append(k)
        lines = [
            f"{name},Title {name},{classes[k]}\n" for name, k in zip(names, labels, strict=True)
        ]
        (titles / f"{split}_titles.csv").write_text("".join(lines), encoding="utf-8")
        y = np.array(labels)
        text = (rng.standard_normal((len(names), LT, D)) + text_centers[y]).astype(np.float16)
        image = (rng.standard_normal((len(names), LI, D)) + image_centers[y]).astype(np.float16)
        mask = np.ones((len(names), LT), dtype=bool)
        mask[:, 4:] = False
        features = root / "features"
        write_features(
            features / "text" / "bert-base-uncased", split, names, text.mean(1), text, mask
        )
        write_features(
            features / "image" / "vit_b_16_swag", split, names, image.mean(1), image,
            np.ones((len(names), LI), dtype=bool),
        )  # fmt: skip
        if split == "train":
            val_ids = [name for name in names if name.endswith(("train0.jpg", "train1.jpg"))]
    (root / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    return val_ids


def compose_cfg(root: Path, *overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(
            config_name="train/cross_attention",
            overrides=[
                f"val_ids={root / 'val_ids.txt'}",
                f"preds_dir={root / 'preds'}",
                f"ckpt_dir={root / 'ckpt'}",
                f"search_dir={root / 'search'}",
                "device=cpu",
                f"model.d_model={TINY.d_model}",
                f"model.heads={TINY.heads}",
                f"model.head_hidden={TINY.head_hidden}",
                "protocol.batch_size=8",
                "protocol.max_epochs=3",
                *overrides,
            ],
        )


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    write_dataset(tmp_path)
    return tmp_path


def test_fit_learns_a_separable_problem(dataset: Path) -> None:
    splits = load_splits(compose_cfg(dataset))
    assert len(splits.val) == 6 and len(splits.train) == 30 and len(splits.test) == 12
    model = CrossAttentionFusion(LT, LI, D, splits.n_classes, TINY)
    result = fit(
        model, splits.train, splits.train_labels, splits.val, splits.val_labels,
        TrainConfig(lr=1e-2, weight_decay=0.0, batch_size=8, max_epochs=20, patience=5),
        torch.device("cpu"), seed=0,
    )  # fmt: skip
    assert result.val_top1 == 1.0
    assert 1 <= result.best_epoch <= 20


def test_ablations_change_only_one_modality(dataset: Path) -> None:
    splits = load_splits(compose_cfg(dataset))
    means = TrainMeans.compute(splits.train)
    assert means.text.shape == (LT, D) and means.image.shape == (LI, D)
    assert means.text_mask.tolist() == [True] * 4 + [False] * 2
    rows = np.arange(len(splits.test))
    both = splits.test.get(rows)
    text_mean = splits.test.ablated("text_mean", means).get(rows)
    assert torch.equal(text_mean["image"], both["image"])
    assert torch.equal(text_mean["text"][0], text_mean["text"][-1])
    shuffled = splits.test.ablated("image_shuffled", means).get(rows)
    assert torch.equal(shuffled["text"], both["text"])
    assert not torch.equal(shuffled["image"], both["image"])
    assert sorted(shuffled["image"].sum((1, 2)).tolist()) == sorted(
        both["image"].sum((1, 2)).tolist()
    )
    with pytest.raises(ValueError, match="unknown condition"):
        splits.test.ablated("nope", means)


def test_run_searches_then_writes_predictions_in_contract_format(dataset: Path) -> None:
    cfg = compose_cfg(dataset, "search.lr=[1e-2]", "search.weight_decay=[0.0]", "seeds=[0,1]")
    run(cfg)
    assert (dataset / "search" / "search.csv").is_file()
    splits = load_splits(cfg)
    for seed in (0, 1):
        folder = dataset / "preds" / "cross_attention" / f"seed{seed}"
        ids, logits = load_preds(folder / "val_both.npz")
        align(ids, logits, splits.val_ids, n_classes=3)
        for condition in CONDITIONS:
            ids, logits = load_preds(folder / f"test_{condition}.npz")
            assert logits.dtype == np.float32
            align(ids, logits, splits.test_ids, n_classes=3)
        assert (dataset / "ckpt" / "cross_attention" / f"seed{seed}.pt").is_file()
    assert (dataset / "preds" / "cross_attention" / "meta.json").is_file()
