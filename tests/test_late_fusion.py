import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.models.late_fusion import (
    CONDITIONS,
    calibrate_temperature,
    choose_weight,
    run,
)
from recipe_fusion.preds import preds_path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CLASSES = ["apple_pie", "bagel", "sushi"]


def _write_predictions(
    outputs: Path,
    method: str,
    seed: int,
    split: str,
    condition: str,
    ids: list[str],
    labels: np.ndarray,
    *,
    offset: int,
) -> None:
    logits = np.full((len(ids), len(CLASSES)), -1, dtype=np.float32)
    logits[np.arange(len(ids)), (labels + offset) % len(CLASSES)] = 2
    path = preds_path(outputs, method, seed, split, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), logits=logits)


def _write_dataset(root: Path) -> tuple[list[str], list[int], list[str], list[int], Path]:
    titles = root / "upmc-food101" / "texts"
    titles.mkdir(parents=True)
    train_lines = [f"{label}_{i}.jpg,title,{label}\n" for label in CLASSES for i in range(3)]
    (titles / "train_titles.csv").write_text("".join(train_lines), encoding="utf-8")
    val_ids = [f"{label}_0.jpg" for label in CLASSES]
    val_file = root / "val_ids.txt"
    val_file.write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    test_ids = [f"{label}_test.jpg" for label in CLASSES]
    (titles / "test_titles.csv").write_text(
        "".join(
            f"{image_id},title,{label}\n" for label, image_id in zip(CLASSES, test_ids, strict=True)
        ),
        encoding="utf-8",
    )
    return val_ids, list(range(3)), test_ids, list(range(3)), val_file


def _config(root: Path, val_file: Path, *overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(
            config_name="late_fusion",
            overrides=[
                f"outputs_dir={root}",
                f"val_ids={val_file}",
                f"paths.data_dir={root}",
                "seeds=[0,1]",
                *overrides,
            ],
        )


def test_temperature_and_weight_grids_select_validation_optima() -> None:
    labels = np.array([0, 1, 0, 1])
    confident = np.array([[4, 0], [0, 4], [4, 0], [0, 4]], dtype=np.float64)
    temperature, nll = calibrate_temperature(confident, labels, np.array([-1.0, 0.0, 1.0]))
    assert temperature == pytest.approx(np.exp(-1))
    assert nll == pytest.approx(-np.log(np.exp(4 / temperature) / (np.exp(4 / temperature) + 1)))

    image = np.array([[5, 0], [0, 5], [5, 0], [0, 5]], dtype=np.float64)
    text = np.array([[0, 4], [4, 0], [0, 4], [4, 0]], dtype=np.float64)
    weight, top1, _ = choose_weight(image, text, labels, 1, 1)
    assert weight == 1
    assert top1 == 1


def test_run_writes_all_fusions_conditions_and_seed_metadata(tmp_path: Path) -> None:
    val_ids, val_labels, test_ids, _, val_file = _write_dataset(tmp_path)
    for seed in range(2):
        for split, ids, labels in (
            ("val", val_ids, np.asarray(val_labels)),
            ("test", test_ids, np.asarray(val_labels)),
        ):
            for condition in ("image_only", "image_mean", "image_shuffled"):
                _write_predictions(
                    tmp_path,
                    "image_mlp",
                    seed,
                    split,
                    condition,
                    ids,
                    labels,
                    offset=0,
                )
            for condition in ("text_only", "text_mean", "text_shuffled"):
                _write_predictions(
                    tmp_path,
                    "text_mlp",
                    seed,
                    split,
                    condition,
                    ids,
                    labels,
                    offset=1,
                )
            for condition in ("text_only", "text_mean", "text_shuffled"):
                _write_predictions(
                    tmp_path,
                    "text_bert_ft",
                    seed,
                    split,
                    condition,
                    ids,
                    labels,
                    offset=0,
                )

    cfg = _config(tmp_path, val_file, "text_ft_method=text_bert_ft")
    summaries = run(cfg)
    assert set(summaries) == {"0", "1"}
    for method in ("late_sum", "late_calibrated", "late_calibrated_ft"):
        for seed in range(2):
            for condition in CONDITIONS:
                for split, ids in (("val", val_ids), ("test", test_ids)):
                    path = preds_path(tmp_path, method, seed, split, condition)
                    with np.load(path, allow_pickle=False) as data:
                        assert data["ids"].tolist() == ids
                        assert data["logits"].shape == (3, 3)
                        assert np.isfinite(data["logits"]).all()

    metadata = json.loads((tmp_path / "preds/late_calibrated/meta.json").read_text())
    assert set(metadata["conditions"]) == {"0", "1"}
    for seed in ("0", "1"):
        assert set(metadata["conditions"][seed]) == set(CONDITIONS)
        for params in metadata["conditions"][seed].values():
            assert -2 <= np.log(params["image_temperature"]) <= 2
            assert -2 <= np.log(params["text_temperature"]) <= 2
            assert 0 <= params["image_weight"] <= 1
            assert params["image_weight"] * 20 == pytest.approx(round(params["image_weight"] * 20))
    ft_meta = json.loads((tmp_path / "preds/late_calibrated_ft/meta.json").read_text())
    assert ft_meta["text_method"] == "text_bert_ft"


def test_missing_ablation_prediction_is_reported(tmp_path: Path) -> None:
    val_ids, val_labels, test_ids, _, val_file = _write_dataset(tmp_path)
    for split, ids, labels in (
        ("val", val_ids, np.asarray(val_labels)),
        ("test", test_ids, np.asarray(val_labels)),
    ):
        _write_predictions(tmp_path, "image_mlp", 0, split, "image_only", ids, labels, offset=0)
        _write_predictions(tmp_path, "text_mlp", 0, split, "text_only", ids, labels, offset=0)
    cfg = _config(tmp_path, val_file, "seeds=[0]")
    with pytest.raises(FileNotFoundError, match="text_mean"):
        run(cfg)
    assert not (tmp_path / "preds/late_sum").exists()


def test_base_fusion_can_use_an_alternate_text_method(tmp_path: Path) -> None:
    val_ids, _, test_ids, _, val_file = _write_dataset(tmp_path)
    for split, ids in (("val", val_ids), ("test", test_ids)):
        labels = np.arange(len(ids))
        _write_predictions(tmp_path, "image_mlp", 0, split, "image_only", ids, labels, offset=0)
        _write_predictions(tmp_path, "tfidf_logreg", 0, split, "text_only", ids, labels, offset=1)
    cfg = _config(
        tmp_path,
        val_file,
        "seeds=[0]",
        "conditions=[both]",
        "text_method=tfidf_logreg",
    )
    run(cfg)
    assert preds_path(tmp_path, "late_sum", 0, "test", "both").is_file()
    assert preds_path(tmp_path, "late_calibrated", 0, "val", "both").is_file()
    assert not (tmp_path / "preds/late_calibrated_ft").exists()
