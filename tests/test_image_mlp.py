from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import Features, write_features
from recipe_fusion.models.image_mlp import ImageMLP
from recipe_fusion.train_image_mlp import _validate_cache, run_training

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FEATURE_META = {
    "encoder": "vit_b_16_swag",
    "preprocessing": "weights.transforms()",
    "pooled": "CLS token after the final LayerNorm",
    "d": 768,
}


def test_image_mlp_has_protocol_2_4_architecture() -> None:
    model = ImageMLP()
    layers = model.layers
    assert isinstance(layers[0], torch.nn.LayerNorm)
    assert layers[0].normalized_shape == (768,)
    assert isinstance(layers[1], torch.nn.Linear)
    assert (layers[1].in_features, layers[1].out_features) == (768, 512)
    assert isinstance(layers[2], torch.nn.GELU)
    assert isinstance(layers[3], torch.nn.Dropout)
    assert layers[3].p == 0.3
    assert isinstance(layers[4], torch.nn.Linear)
    assert (layers[4].in_features, layers[4].out_features) == (512, 101)
    assert model(torch.zeros(2, 768)).shape == (2, 101)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("preprocessing", "wrong transform", "preprocessing"),
        ("encoder", "other_encoder", "encoder"),
        ("pooled", "wrong output", "pooled"),
        ("d", 512, "'d'"),
    ],
)
def test_preflight_rejects_mismatched_feature_metadata(field, value, message) -> None:
    ids = ["a.jpg"]
    metadata = {**FEATURE_META, field: value}
    features = Features(ids, np.zeros((1, 768), dtype=np.float16), meta=metadata)
    with pytest.raises(ValueError, match=message):
        _validate_cache(
            features,
            [Sample("a.jpg", "title", "dish", "train")],
            "train",
            expected_encoder="vit_b_16_swag",
        )


def test_preflight_rejects_ids_that_do_not_match_title_order() -> None:
    ids = ["b.jpg", "a.jpg"]
    features = Features(ids, np.zeros((2, 768), dtype=np.float16), meta=FEATURE_META)
    samples = [
        Sample("a.jpg", "title", "dish", "train"),
        Sample("b.jpg", "title", "dish", "train"),
    ]
    with pytest.raises(ValueError, match="order_matches=False"):
        _validate_cache(features, samples, "train", expected_encoder="vit_b_16_swag")


def test_training_saves_checkpoint_and_validation_test_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    data_dir = tmp_path / "upmc-food101"
    texts = data_dir / "texts"
    texts.mkdir(parents=True)

    train_ids = [f"class_{label:03d}_{part}.jpg" for label in range(101) for part in range(2)]
    (texts / "train_titles.csv").write_text(
        "".join(
            f"{image_id},title,class_{label:03d}\n"
            for label in range(101)
            for image_id in (f"class_{label:03d}_0.jpg", f"class_{label:03d}_1.jpg")
        ),
        encoding="utf-8",
    )
    test_ids = [f"class_{label:03d}_test.jpg" for label in range(101)]
    (texts / "test_titles.csv").write_text(
        "".join(f"{image_id},title,class_{label:03d}\n" for label, image_id in enumerate(test_ids)),
        encoding="utf-8",
    )
    val_ids = [f"class_{label:03d}_0.jpg" for label in range(101)]
    val_file = tmp_path / "val_ids.txt"
    val_file.write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    feature_dir = tmp_path / "features/image/vit_b_16_swag"
    rng = np.random.default_rng(4)
    write_features(
        feature_dir,
        "train",
        train_ids,
        rng.normal(size=(len(train_ids), 768)).astype(np.float16),
        meta=FEATURE_META,
    )
    write_features(
        feature_dir,
        "test",
        test_ids,
        rng.normal(size=(len(test_ids), 768)).astype(np.float16),
        meta=FEATURE_META,
    )

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            config_name="image_mlp",
            overrides=[
                f"val_ids={val_file}",
                f"checkpoint_dir={tmp_path / 'ckpt'}",
                f"preds_dir={tmp_path / 'preds'}",
                "train.seeds=[3]",
                "train.batch_size=101",
                "train.epochs=1",
                "train.patience=1",
                "train.device=cpu",
            ],
        )

    result = run_training(cfg)
    assert result[0]["seed"] == 3
    assert (tmp_path / "ckpt/seed3.pt").is_file()
    checkpoint = torch.load(tmp_path / "ckpt/seed3.pt", map_location="cpu", weights_only=True)
    assert checkpoint["classes"] == [f"class_{label:03d}" for label in range(101)]
    assert checkpoint["seed"] == 3
    assert checkpoint["best_val_top1"] == result[0]["best_val_top1"]

    for split, expected_ids in (("val", val_ids), ("test", test_ids)):
        with np.load(tmp_path / f"preds/image_mlp/seed3/{split}_image.npz") as predictions:
            assert predictions["ids"].tolist() == expected_ids
            assert predictions["logits"].shape == (len(expected_ids), 101)
            assert np.isfinite(predictions["logits"]).all()
