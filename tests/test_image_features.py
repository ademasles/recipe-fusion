import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import Tensor, nn

from recipe_fusion.data.upmc import read_split
from recipe_fusion.features.cache import read_features
from recipe_fusion.features.extract_image import extract_split
from recipe_fusion.features.image import ImageIssue, check_images, write_report

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 768)

    def forward(self, images: Tensor) -> Tensor:
        pooled = self.projection(images.mean(dim=(2, 3)))
        return pooled[:, None, :].expand(-1, 577, -1)


class TinyImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TinyEncoder()
        self.heads = nn.Identity()

    def forward(self, images: Tensor) -> Tensor:
        return self.heads(self.encoder(images)[:, 0, :])


def test_check_images_finds_truncated_and_missing_images(tmp_path: Path) -> None:
    valid_path = tmp_path / "images/train/dish/valid.jpg"
    truncated_path = tmp_path / "images/train/dish/truncated.jpg"
    valid_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (16, 16), color="red")
    image.save(valid_path, format="JPEG")
    image_data = valid_path.read_bytes()
    truncated_path.write_bytes(image_data[: len(image_data) // 2])

    titles = tmp_path / "texts/train_titles.csv"
    titles.parent.mkdir(parents=True)
    titles.write_text(
        "valid.jpg,Valid,dish\ntruncated.jpg,Truncated,dish\nmissing.jpg,Missing,dish\n",
        encoding="utf-8",
    )
    issues = check_images(tmp_path, splits=("train",))

    assert len(issues) == 2
    assert {issue.path.name: issue.status for issue in issues} == {
        "truncated.jpg": "unreadable",
        "missing.jpg": "missing",
    }


def test_write_report_creates_csv(tmp_path: Path) -> None:
    report = tmp_path / "image_issues.csv"
    write_report([ImageIssue(tmp_path / "missing.jpg", "missing", "not found")], report)

    with report.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["image", "status", "error"],
        [str(tmp_path / "missing.jpg"), "missing", "not found"],
    ]


def test_extract_split_writes_pooled_and_grid_features_on_cpu(tmp_path: Path) -> None:
    image_directory = tmp_path / "images/train/dish"
    image_directory.mkdir(parents=True)
    title_lines = []
    for index in range(3):
        image = Image.new("L" if index == 1 else "RGB", (16, 16), color=index * 40)
        name = f"dish_{index}.jpg"
        image.save(image_directory / name, format="JPEG")
        title_lines.append(f"{name},Dish {index},dish")

    titles = tmp_path / "texts/train_titles.csv"
    titles.parent.mkdir(parents=True)
    titles.write_text("\n".join(title_lines) + "\n", encoding="utf-8")
    samples = read_split(tmp_path, "train")

    def to_tensor(image: Image.Image) -> Tensor:
        assert image.mode == "RGB"
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255

    output = tmp_path / "features/image/vit_b_16_swag"
    extract_split(
        tmp_path,
        samples,
        TinyImageModel(),
        to_tensor,
        output,
        "train",
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    features = read_features(output, "train", mmap=False)
    assert features.ids == [sample.image_name for sample in samples]
    assert features.pooled.shape == (3, 768)
    assert features.pooled.dtype == np.float16
    assert features.tokens is not None
    assert features.tokens.shape == (3, 64, 768)
    assert features.tokens.dtype == np.float16
    assert features.mask is not None
    assert features.mask.shape == (3, 64)
    assert features.mask.dtype == np.bool_
    assert features.mask.all()
    assert features.meta["format"] == "2.2"
    assert features.meta["extraction_seconds"] >= 0
    assert (features.meta["d"], features.meta["L"]) == (768, 64)
    assert np.isfinite(features.pooled).all()
    assert np.isfinite(features.tokens).all()


def test_extract_config_follows_f101_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="features/image")
    assert cfg.paths.upmc_dir == f"{tmp_path}/upmc-food101"
    assert cfg.paths.features_dir == f"{tmp_path}/features"
    assert cfg.extract.batch_size == 64
