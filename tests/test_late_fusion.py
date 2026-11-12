import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.data.loading import load_splits
from recipe_fusion.features.cache import write_features
from recipe_fusion.models.heads import MLPHead, save_head
from recipe_fusion.models.late_fusion import (
    CONDITIONS,
    Fusion,
    calibrate_temperature,
    choose_weight,
    mean_row,
    run,
)
from recipe_fusion.preds import ckpt_path, preds_path, write_preds

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CLASSES = ["apple_pie", "bagel", "sushi"]
DIM = 4


def test_temperature_and_weight_grids_select_validation_optima() -> None:
    labels = np.array([0, 1, 0, 1])
    confident = np.array([[4, 0], [0, 4], [4, 0], [0, 4]], dtype=np.float64)
    temperature, _ = calibrate_temperature(confident, labels, np.array([-1.0, 0.0, 1.0]))
    assert temperature == pytest.approx(np.exp(-1))

    image = np.array([[5, 0], [0, 5], [5, 0], [0, 5]], dtype=np.float64)
    text = np.array([[0, 4], [4, 0], [0, 4], [4, 0]], dtype=np.float64)
    weight, top1, _ = choose_weight(image, text, labels, 1, 1)
    assert (weight, top1) == (1, 1)


def write_dataset(root: Path) -> Path:
    titles = root / "upmc-food101" / "texts"
    titles.mkdir(parents=True)
    train = [f"{c}_{i}.jpg,title,{c}\n" for c in CLASSES for i in range(4)]
    test = [f"{c}_t{i}.jpg,title,{c}\n" for c in CLASSES for i in range(3)]
    (titles / "train_titles.csv").write_text("".join(train))
    (titles / "test_titles.csv").write_text("".join(test))
    val_file = root / "val_ids.txt"
    val_file.write_text("".join(f"{c}_{i}.jpg\n" for c in CLASSES for i in (0, 1)))
    return val_file


def write_unimodal(outputs: Path, method: str, condition: str, ids: list[str], seed: int) -> None:
    rng = np.random.default_rng(hash((method, seed)) % 2**32)
    labels = np.array([CLASSES.index(i.rsplit("_", 1)[0]) for i in ids])
    logits = rng.normal(size=(len(ids), len(CLASSES))).astype(np.float32)
    logits[np.arange(len(ids)), labels] += 1.5
    split = "test" if "_t" in ids[0] else "val"
    write_preds(preds_path(outputs, method, seed, split, condition), ids, logits, n_classes=3)


def setup(tmp_path: Path, *, text_head: bool):
    val_file = write_dataset(tmp_path)
    splits = load_splits(tmp_path / "upmc-food101", val_file)
    val_ids = [s.image_name for s in splits.val]
    test_ids = [s.image_name for s in splits.test]
    outputs = tmp_path / "outputs"
    for seed in (0, 1):
        for ids in (val_ids, test_ids):
            write_unimodal(outputs, "image_mlp", "image_only", ids, seed)
            write_unimodal(outputs, "text_mlp", "text_only", ids, seed)
    features = {}
    for modality in ("image", "text") if text_head else ("image",):
        method = f"{modality}_mlp"
        features[modality] = tmp_path / "features" / modality
        pooled = np.random.default_rng(0).normal(size=(len(splits.train_all), DIM))
        write_features(
            features[modality],
            "train",
            [s.image_name for s in splits.train_all],
            pooled.astype(np.float16),
        )
        for seed in (0, 1):
            head = MLPHead(dim_in=DIM, hidden=8, n_classes=3, dropout=0.0)
            save_head(ckpt_path(outputs, method, seed), head, seed=seed)
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            config_name="train/late_fusion",
            overrides=[
                f"outputs_dir={outputs}",
                "seeds=[0,1]",
                f"image.features={tmp_path / 'features' / 'image'}",
                f"text.features={tmp_path / 'features' / 'text'}",
            ],
        )
    return cfg, splits, outputs, test_ids


def load(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path) as data:
        return data["ids"].tolist(), data["logits"]


def test_fusion_is_fitted_once_and_frozen_for_the_ablations(tmp_path: Path) -> None:
    cfg, splits, outputs, test_ids = setup(tmp_path, text_head=True)
    meta = run(cfg, splits)

    for seed in (0, 1):
        fusion = Fusion(
            **{
                k: meta["seeds"][str(seed)][k]
                for k in ("image_temperature", "text_temperature", "image_weight")
            }
        )
        _, image = load(preds_path(outputs, "image_mlp", seed, "test", "image_only"))
        _, text = load(preds_path(outputs, "text_mlp", seed, "test", "text_only"))
        permutation = np.random.default_rng(0).permutation(len(test_ids))
        expected = {
            "both": fusion(image, text),
            "image_shuffled": fusion(image[permutation], text),
            "text_shuffled": fusion(image, text[permutation]),
        }
        train_ids = [s.image_name for s in splits.train]
        for modality in ("image", "text"):
            row = mean_row(
                outputs, f"{modality}_mlp", seed, tmp_path / "features" / modality, train_ids
            )
            assert row is not None
            constant = np.broadcast_to(row, image.shape)
            expected[f"{modality}_mean"] = (
                fusion(constant, text) if modality == "image" else fusion(image, constant)
            )
        for condition in CONDITIONS:
            ids, logits = load(preds_path(outputs, "late_calibrated", seed, "test", condition))
            assert ids == test_ids
            np.testing.assert_allclose(logits, expected[condition], rtol=1e-6)
            assert preds_path(outputs, "late_sum", seed, "test", condition).is_file()
        assert preds_path(outputs, "late_calibrated", seed, "val", "both").is_file()

    saved = json.loads((outputs / "preds" / "late_calibrated" / "meta.json").read_text())
    assert set(saved["seeds"]) == {"0", "1"}
    assert saved["text"]["method"] == "text_mlp"


def test_mean_ablation_is_skipped_without_a_head(tmp_path: Path) -> None:
    cfg, splits, outputs, _ = setup(tmp_path, text_head=False)
    run(cfg, splits)
    assert preds_path(outputs, "late_calibrated", 0, "test", "image_mean").is_file()
    assert not preds_path(outputs, "late_calibrated", 0, "test", "text_mean").exists()
    assert preds_path(outputs, "late_calibrated", 0, "test", "text_shuffled").is_file()


def test_missing_predictions_are_reported(tmp_path: Path) -> None:
    cfg, splits, _, _ = setup(tmp_path, text_head=True)
    cfg.text.method = "tfidf_logreg"
    with pytest.raises(FileNotFoundError, match="tfidf_logreg"):
        run(cfg, splits)
