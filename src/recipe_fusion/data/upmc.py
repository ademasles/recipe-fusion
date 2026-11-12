"""UPMC Food-101 index: (image, page title, class) triples per split.

Layout of the Kaggle copy (``gianmarco96/upmcfood101``, version 1)::

    texts/{train,test}_titles.csv         # no header: image file name, page title, class
    images/{train,test}/<class>/<file>.jpg

Titles are kept exactly as they are stored: they still contain HTML entities such as
``&#8217;``. Every text model cleans them with :func:`recipe_fusion.data.clean.clean_title`.
"""

import csv
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "test")


@dataclass(frozen=True)
class Sample:
    image_name: str
    text: str
    label: str
    split: str

    def image_path(self, root: Path) -> Path:
        return root / "images" / self.split / self.label / self.image_name


def titles_path(root: Path, split: str) -> Path:
    return root / "texts" / f"{split}_titles.csv"


def read_split(root: Path, split: str) -> list[Sample]:
    """Read ``texts/<split>_titles.csv``; raise on rows that do not have three fields."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    samples = []
    with titles_path(root, split).open(encoding="utf-8", newline="") as f:
        for line, row in enumerate(csv.reader(f), start=1):
            if len(row) != 3:
                raise ValueError(f"{split}_titles.csv line {line}: expected 3 fields, got {row!r}")
            image_name, text, label = row
            samples.append(Sample(image_name, text, label, split))
    return samples


def class_names(samples: Iterable[Sample]) -> list[str]:
    """Sorted class names; the index in this list is the integer label."""
    return sorted({sample.label for sample in samples})


def missing_images(root: Path, samples: Iterable[Sample]) -> list[Sample]:
    """Samples whose image file is not on disk."""
    return [sample for sample in samples if not sample.image_path(root).is_file()]


def summary(samples: Iterable[Sample]) -> dict[str, int]:
    """Number of samples and of classes, and the smallest and largest class sizes."""
    counts = Counter(sample.label for sample in samples)
    if not counts:
        return {"samples": 0, "classes": 0, "min_per_class": 0, "max_per_class": 0}
    return {
        "samples": sum(counts.values()),
        "classes": len(counts),
        "min_per_class": min(counts.values()),
        "max_per_class": max(counts.values()),
    }
