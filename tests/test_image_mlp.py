from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("config", "method", "condition", "features"),
    [
        ("text_mlp", "text_mlp", "text_only", "text/bert-base-uncased"),
        ("image_mlp", "image_mlp", "image_only", "image/vit_b_16_swag"),
    ],
)
def test_pooled_mlp_configs_differ_only_by_modality(
    config: str, method: str, condition: str, features: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F101_DATA_DIR", "/data")
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name=f"train/{config}")
        reference = compose(config_name="train/text_mlp")
    assert (cfg.method, cfg.condition) == (method, condition)
    assert cfg.features == f"/data/features/{features}"
    for key in ("sweep", "head", "train", "seeds", "val_ids"):
        assert cfg[key] == reference[key]
