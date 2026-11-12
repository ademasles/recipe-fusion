from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import write_features
from recipe_fusion.models import pooled_mlp, tfidf_logreg
from recipe_fusion.models.heads import load_head
from recipe_fusion.preds import ckpt_path, preds_path

CLASSES = ["apple_pie", "bagel", "caesar_salad"]
WORDS = ["pommes", "bretzel", "romaine"]  # one signature word per class


def write_titles(root: Path, split: str, rows: list[tuple[str, str, str]]) -> None:
    path = root / "texts" / f"{split}_titles.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n},{t},{c}\n" for n, t, c in rows), encoding="utf-8")


def make_samples(split: str, indices: range) -> list[Sample]:
    return [
        Sample(f"{label}_{i}.jpg", f"{word} tasty dish {i}", label, split)
        for label, word in zip(CLASSES, WORDS, strict=True)
        for i in indices
    ]


def make_splits() -> Splits:
    train, val = make_samples("train", range(3, 12)), make_samples("train", range(3))
    return Splits(train + val, train, val, make_samples("test", range(12, 16)), CLASSES)


def test_load_splits_removes_validation_from_train(tmp_path) -> None:
    train = [(f"a_{i}.jpg", f"apple {i}", "apple_pie") for i in range(4)]
    train += [(f"b_{i}.jpg", f"bagel {i}", "bagel") for i in range(4)]
    write_titles(tmp_path, "train", train)
    write_titles(tmp_path, "test", [("a_9.jpg", "apple", "apple_pie")])
    val_ids = tmp_path / "val_ids.txt"
    val_ids.write_text("a_0.jpg\nb_0.jpg\n", encoding="utf-8")

    splits = load_splits(tmp_path, val_ids)
    assert [s.image_name for s in splits.val] == ["a_0.jpg", "b_0.jpg"]
    assert len(splits.train) == 6 and len(splits.train_all) == 8
    assert splits.classes == ["apple_pie", "bagel"]
    assert [s.image_name for s in splits.test] == ["a_9.jpg"]


def test_read_aligned_checks_ids_and_rows_follow_requested_order(tmp_path) -> None:
    samples = [Sample(n, "t", "a", "train") for n in ("x.jpg", "y.jpg", "z.jpg")]
    pooled = np.arange(12, dtype=np.float16).reshape(3, 4)
    write_features(tmp_path, "train", [s.image_name for s in samples], pooled)

    features = pooled_mlp.read_aligned(tmp_path, "train", samples)
    np.testing.assert_array_equal(pooled_mlp.rows(features, ["z.jpg", "x.jpg"]), pooled[[2, 0]])
    with pytest.raises(ValueError, match="ids"):
        pooled_mlp.read_aligned(tmp_path, "train", samples[::-1])


def test_best_config_prefers_top1_then_lower_loss() -> None:
    trials = [
        {"val_top1": 0.8, "val_loss": 0.9},
        {"val_top1": 0.9, "val_loss": 0.7},
        {"val_top1": 0.9, "val_loss": 0.5},
    ]
    assert pooled_mlp.best_config(trials) is trials[2]


def test_tfidf_logreg_writes_the_same_preds_for_every_seed(tmp_path) -> None:
    splits = make_splits()
    cfg = DictConfig(
        {
            "method": "tfidf_logreg",
            "condition": "text_only",
            "outputs_dir": str(tmp_path),
            "seeds": [0, 1, 2],
            "c_grid": [1.0, 10.0],
            "max_iter": 200,
        }
    )
    summary = tfidf_logreg.run(cfg, splits)
    assert summary["chosen"]["val_top1"] == 1.0

    val = [np.load(preds_path(tmp_path, "tfidf_logreg", k, "val", "text_only")) for k in range(3)]
    assert val[0]["ids"].tolist() == [s.image_name for s in splits.val]
    assert val[0]["logits"].shape == (len(splits.val), 3)
    np.testing.assert_array_equal(val[0]["logits"], val[2]["logits"])
    test = np.load(preds_path(tmp_path, "tfidf_logreg", 0, "test", "text_only"))
    assert test["logits"].shape == (len(splits.test), 3)


def make_text_data() -> pooled_mlp.PooledData:
    rng = np.random.default_rng(0)

    def block(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        y = np.repeat(np.arange(3), n_per_class)
        x = rng.standard_normal((len(y), 8)).astype(np.float32)
        x[np.arange(len(y)), y] += 4.0
        return x.astype(np.float16), y

    x_train, y_train = block(30)
    x_val, y_val = block(10)
    x_test, _ = block(5)
    return pooled_mlp.PooledData(
        classes=CLASSES,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        ids_val=[f"v{i}.jpg" for i in range(len(y_val))],
        x_test=x_test,
        ids_test=[f"t{i}.jpg" for i in range(len(x_test))],
    )


def test_text_mlp_sweeps_then_writes_final_outputs(tmp_path) -> None:
    cfg = DictConfig(
        {
            "method": "text_mlp",
            "condition": "text_only",
            "outputs_dir": str(tmp_path),
            "seeds": [0, 1],
            "sweep": {"seed": 0, "lr": [0.001, 0.01], "weight_decay": [0.0001, 0.01]},
            "head": {"hidden": 16, "dropout": 0.1},
            "train": {"batch_size": 32, "max_epochs": 5, "patience": 2},
        }
    )
    summary = pooled_mlp.run(cfg, make_text_data(), torch.device("cpu"))
    assert len(summary["trials"]) == 4
    assert summary["chosen"] in summary["trials"]
    for seed in (0, 1):
        for split in ("val", "test"):
            assert preds_path(tmp_path, "text_mlp", seed, split, "text_only").is_file()
        assert load_head(ckpt_path(tmp_path, "text_mlp", seed)).args["n_classes"] == 3
    assert summary["val_top1_std"] >= 0.0
