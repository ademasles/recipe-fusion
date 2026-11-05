from collections import Counter
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.data.split import (
    make_split,
    make_val_split,
    read_ids,
    split_train_val,
    write_ids,
)
from recipe_fusion.data.upmc import Sample, titles_path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SIZES = {"apple_pie": 40, "pho": 25, "sushi": 7}


def make_samples() -> list[Sample]:
    return [
        Sample(f"{label}_{i}.jpg", f"title {i}", label, "train")
        for label, n in SIZES.items()
        for i in range(n)
    ]


def test_split_is_disjoint_and_complete() -> None:
    samples = make_samples()
    val_ids = make_val_split(samples)
    train, val = split_train_val(samples, val_ids)
    assert {s.image_name for s in train}.isdisjoint(s.image_name for s in val)
    assert len(train) + len(val) == len(samples)
    assert sorted(s.image_name for s in val) == val_ids


def test_split_is_stratified() -> None:
    val_ids = make_val_split(make_samples(), frac=0.2)
    counts = Counter(name.rsplit("_", 1)[0] for name in val_ids)
    assert set(counts) == set(SIZES)
    for label, n in SIZES.items():
        assert abs(counts[label] - 0.2 * n) <= 1


def test_split_keeps_one_sample_on_each_side() -> None:
    samples = [Sample(f"a_{i}.jpg", "", "a", "train") for i in range(2)]
    assert len(make_val_split(samples, frac=0.9)) == 1
    assert len(make_val_split(samples, frac=0.01)) == 1


def test_split_depends_on_seed_not_on_order() -> None:
    samples = make_samples()
    assert make_val_split(samples, seed=0) == make_val_split(samples[::-1], seed=0)
    assert make_val_split(samples, seed=0) != make_val_split(samples, seed=1)


@pytest.mark.parametrize(
    ("samples", "frac", "match"),
    [
        (make_samples(), 0.0, "frac"),
        (make_samples(), 1.0, "frac"),
        (make_samples() + make_samples()[:1], 0.1, "unique"),
        ([Sample("a_0.jpg", "", "a", "train")], 0.1, "fewer than 2"),
    ],
)
def test_invalid_inputs_are_rejected(samples, frac, match) -> None:
    with pytest.raises(ValueError, match=match):
        make_val_split(samples, frac=frac)


def test_unknown_val_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="not in the samples"):
        split_train_val(make_samples(), ["missing.jpg"])


def test_ids_file_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "val_ids.txt"
    write_ids(path, ["a.jpg", "b.jpg"])
    assert read_ids(path) == ["a.jpg", "b.jpg"]
    with pytest.raises(FileExistsError):
        write_ids(path, ["c.jpg"])
    assert read_ids(path) == ["a.jpg", "b.jpg"]


def test_make_split_from_title_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    titles = titles_path(tmp_path / "upmc-food101", "train")
    titles.parent.mkdir(parents=True)
    titles.write_text(
        "".join(f"{s.image_name},{s.text},{s.label}\n" for s in make_samples()), encoding="utf-8"
    )
    out = tmp_path / "val_ids.txt"
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="split", overrides=[f"split.val_ids={out}"])
    assert make_split(cfg) == out
    assert read_ids(out) == make_val_split(make_samples())
