from pathlib import Path

import pytest

from recipe_fusion.data.upmc import (
    class_names,
    missing_images,
    read_split,
    summary,
    titles_path,
)


def write_titles(root: Path, split: str, lines: list[str]) -> None:
    path = titles_path(root, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_split_keeps_commas_inside_quoted_titles(tmp_path: Path) -> None:
    write_titles(tmp_path, "train", ['apple_pie_1.jpg,"Pie, apple & cinnamon",apple_pie'])
    [sample] = read_split(tmp_path, "train")
    assert sample.text == "Pie, apple & cinnamon"
    assert sample.label == "apple_pie"
    assert sample.image_path(tmp_path) == tmp_path / "images/train/apple_pie/apple_pie_1.jpg"


def test_read_split_keeps_html_entities(tmp_path: Path) -> None:
    write_titles(tmp_path, "test", ["pho_3.jpg,Cookin&#8217; Pho,pho"])
    assert read_split(tmp_path, "test")[0].text == "Cookin&#8217; Pho"


def test_read_split_rejects_malformed_rows(tmp_path: Path) -> None:
    write_titles(tmp_path, "train", ["pho_1.jpg,Pho,pho", "pho_2.jpg,pho"])
    with pytest.raises(ValueError, match="line 2"):
        read_split(tmp_path, "train")


def test_read_split_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_split(tmp_path, "val")


def test_class_names_and_summary(tmp_path: Path) -> None:
    write_titles(tmp_path, "train", ["b_1.jpg,B,b", "a_1.jpg,A,a", "a_2.jpg,A2,a"])
    samples = read_split(tmp_path, "train")
    assert class_names(samples) == ["a", "b"]
    assert summary(samples) == {"samples": 3, "classes": 2, "min_per_class": 1, "max_per_class": 2}
    assert summary([])["samples"] == 0


def test_missing_images(tmp_path: Path) -> None:
    write_titles(tmp_path, "train", ["a_1.jpg,A,a", "a_2.jpg,A2,a"])
    samples = read_split(tmp_path, "train")
    samples[0].image_path(tmp_path).parent.mkdir(parents=True)
    samples[0].image_path(tmp_path).write_bytes(b"")
    assert missing_images(tmp_path, samples) == [samples[1]]
