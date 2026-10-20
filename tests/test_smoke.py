"""Smoke tests: the configuration composes and the download step runs without the network."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.data import download as download_module

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def compose_download(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name="download", overrides=list(overrides))


def test_download_config_composes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    cfg = compose_download()
    assert cfg.data.kaggle_handle == "gianmarco96/upmcfood101"
    assert cfg.paths.upmc_dir == f"{tmp_path}/upmc-food101"


def test_download_texts_only_with_fake_kagglehub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_dataset_download(handle: str, path: str | None = None, *, output_dir: str) -> str:
        calls.append((handle, path))
        target = Path(output_dir) / str(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("pho_1.jpg,Pho,pho\n", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(download_module.kagglehub, "dataset_download", fake_dataset_download)
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    root = download_module.download(compose_download("data.texts_only=true"))
    assert root == tmp_path / "upmc-food101"
    assert calls == [
        ("gianmarco96/upmcfood101/versions/1", "texts/train_titles.csv"),
        ("gianmarco96/upmcfood101/versions/1", "texts/test_titles.csv"),
    ]
