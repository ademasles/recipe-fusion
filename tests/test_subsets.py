import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.data.clean import title_key
from recipe_fusion.data.split import read_ids
from recipe_fusion.data.subsets import (
    generic_titles,
    image_duplicates,
    image_subsets,
    lookup_accuracy,
    make_subsets,
    subsets,
    title_labels,
)
from recipe_fusion.data.upmc import Sample

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

TRAIN = [
    Sample("t0.jpg", "Apple Pie Recipe", "apple_pie", "train"),
    Sample("t1.jpg", "apple pie recipe", "apple_pie", "train"),
    Sample("t2.jpg", "Grandma&amp;#039;s Favourite", "apple_pie", "train"),
    Sample("t3.jpg", "Grandma's favourite", "bagel", "train"),
    Sample("t4.jpg", "Grandma's favourite", "bagel", "train"),
]


def test_title_key_folds_case_accents_and_entities() -> None:
    assert title_key("Crème  BRÛLÉE &amp;amp; co") == title_key("creme brulee & co")


def test_subsets_split_seen_unseen_and_class_name() -> None:
    samples = [
        Sample("a.jpg", "APPLE PIE RECIPE", "apple_pie", "test"),  # seen, names its class
        Sample("b.jpg", "Grandma's Favourite", "apple_pie", "test"),  # seen, no class name
        Sample("c.jpg", "Bagels at dawn", "bagel", "test"),  # unseen, names its class
    ]
    out = subsets(samples, title_labels(TRAIN))
    assert out == {
        "title_seen": ["a.jpg", "b.jpg"],
        "title_unseen": ["c.jpg"],
        "no_class_name": ["b.jpg"],
    }


def test_lookup_accuracy_uses_the_majority_training_label() -> None:
    labels = title_labels(TRAIN)
    samples = [
        Sample("a.jpg", "Apple pie recipe", "apple_pie", "test"),  # apple_pie 2/2: hit
        Sample("b.jpg", "Grandma's favourite", "apple_pie", "test"),  # bagel 2 vs 1: miss
        Sample("c.jpg", "Never seen", "bagel", "test"),  # ignored
    ]
    assert lookup_accuracy(samples, labels) == 0.5
    assert lookup_accuracy(samples[2:], labels) != lookup_accuracy(samples[2:], labels)  # NaN


def test_generic_titles_rank_by_count() -> None:
    top = generic_titles(title_labels(TRAIN), top=1)
    assert top[0]["count"] == 3 and top[0]["classes"] == 2


def write_image(root: Path, sample: Sample, payload: bytes) -> None:
    path = sample.image_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_image_duplicates_finds_identical_bytes(tmp_path: Path) -> None:
    train = TRAIN[:2]
    test = [Sample("x.jpg", "x", "bagel", "test"), Sample("y.jpg", "y", "bagel", "test")]
    for i, sample in enumerate(train):
        write_image(tmp_path, sample, f"img{i}".encode())
    write_image(tmp_path, test[0], b"img1")
    write_image(tmp_path, test[1], b"other")
    assert image_duplicates(tmp_path, train, test) == [
        {"id": "x.jpg", "train_id": "t1.jpg", "same_label": False}
    ]


def write_titles(root: Path, split: str, samples: list[Sample]) -> None:
    path = root / "upmc-food101" / "texts" / f"{split}_titles.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{s.image_name},{s.text},{s.label}\n" for s in samples))


def test_make_subsets_writes_files_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    val = Sample("v.jpg", "Apple pie recipe", "apple_pie", "train")
    test = [
        Sample("a.jpg", "apple pie recipe", "apple_pie", "test"),
        Sample("b.jpg", "Something new", "bagel", "test"),
    ]
    write_titles(tmp_path, "train", [*TRAIN, val])
    write_titles(tmp_path, "test", test)
    root = tmp_path / "upmc-food101"
    for i, sample in enumerate([*TRAIN, val, *test]):
        write_image(root, sample, b"same" if sample.image_name == "b.jpg" else f"{i}".encode())
    write_image(root, test[1], TRAIN[3].image_path(root).read_bytes())
    (tmp_path / "val_ids.txt").write_text("v.jpg\n")

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            config_name="subsets",
            overrides=[
                f"val_ids={tmp_path / 'val_ids.txt'}",
                f"out_dir={tmp_path / 'splits'}",
                f"report={tmp_path / 'report.json'}",
            ],
        )
    make_subsets(cfg)

    splits = tmp_path / "splits"
    assert read_ids(splits / "val_title_seen.txt") == ["v.jpg"]
    assert read_ids(splits / "test_title_seen.txt") == ["a.jpg"]
    assert read_ids(splits / "test_title_unseen.txt") == ["b.jpg"]
    assert read_ids(splits / "test_no_class_name.txt") == ["b.jpg"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["test"]["lookup_top1_on_title_seen"] == 1.0
    assert report["test_image_duplicates"]["n"] == 1
    assert read_ids(splits / "test_image_seen.txt") == ["b.jpg"]
    assert read_ids(splits / "test_image_unseen.txt") == ["a.jpg"]
    assert read_ids(splits / "test_unseen.txt") == []
    assert report["train_titles"]["distinct"] == 2


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluate_config_points_to_the_split_subsets(split: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="evaluate", overrides=[f"split={split}"])
    assert dict(cfg.subsets) == {
        name: f"splits/{split}_{name}.txt"
        for name in ("title_seen", "title_unseen", "no_class_name", "image_unseen", "unseen")
    }


def test_image_subsets_combine_title_and_image() -> None:
    samples = [Sample(f"{c}.jpg", c, "bagel", "test") for c in "abc"]
    members = {"title_unseen": ["b.jpg", "c.jpg"]}
    assert image_subsets(samples, members, {"c.jpg"}) == {
        "image_seen": ["c.jpg"],
        "image_unseen": ["a.jpg", "b.jpg"],
        "unseen": ["b.jpg"],
    }
