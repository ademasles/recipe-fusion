"""Samples used by the text baselines: train without validation, validation and test."""

from dataclasses import dataclass
from pathlib import Path

from recipe_fusion.data.split import read_ids, split_train_val
from recipe_fusion.data.upmc import Sample, class_names, read_split


@dataclass(frozen=True)
class Splits:
    train_all: list[Sample]  # official training set, in file order
    train: list[Sample]  # training set without the validation subset
    val: list[Sample]
    test: list[Sample]
    classes: list[str]  # index in this list = integer label


def load_splits(upmc_dir: Path, val_ids_path: Path) -> Splits:
    train_all = read_split(upmc_dir, "train")
    train, val = split_train_val(train_all, read_ids(val_ids_path))
    return Splits(train_all, train, val, read_split(upmc_dir, "test"), class_names(train_all))
