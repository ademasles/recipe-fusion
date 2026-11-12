"""Evaluation subsets and the duplicate-title analysis.

Two facts make the overall accuracy hard to read (``docs/plan.md``): most titles spell out the
class name, and many test titles appear verbatim in the training set. This module writes the
subsets that separate those cases, so every method is also scored where the title alone is not
enough:

- ``title_seen`` / ``title_unseen``: the title (case and accents folded) does / does not occur
  in the training part the models are fitted on (official train minus validation);
- ``no_class_name``: the title does not contain the class name;
- ``image_seen`` / ``image_unseen``: the image file is / is not byte-identical to a training
  image (``check_images: true``);
- ``unseen``: neither the title nor the image occurs in training.

Run with ``just make-subsets``. The id files land in ``splits/`` and are committed; they depend
only on the titles and the frozen validation split. The report (counts, how often a seen title
keeps its training label, generic titles, byte-identical images across splits) goes to
``outputs/subsets/``.
"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.clean import mentions_class, title_key
from recipe_fusion.data.split import read_ids, split_train_val, write_ids
from recipe_fusion.data.upmc import Sample, read_split

log = logging.getLogger(__name__)

SUBSETS = ("title_seen", "title_unseen", "no_class_name")


def title_labels(train: Iterable[Sample]) -> dict[str, Counter[str]]:
    """Folded title -> how many training samples carry it, per class."""
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in train:
        labels[title_key(sample.text)][sample.label] += 1
    return dict(labels)


def subsets(samples: Sequence[Sample], seen: Mapping[str, object]) -> dict[str, list[str]]:
    """Image names of each subset, in the order of ``samples``."""
    out: dict[str, list[str]] = {name: [] for name in SUBSETS}
    for sample in samples:
        is_seen = title_key(sample.text) in seen
        out["title_seen" if is_seen else "title_unseen"].append(sample.image_name)
        if not mentions_class(sample.text, sample.label):
            out["no_class_name"].append(sample.image_name)
    return out


def lookup_accuracy(samples: Sequence[Sample], labels: Mapping[str, Counter[str]]) -> float:
    """Top-1 of "predict the majority training label of the same title" on seen titles.

    Ties go to the alphabetically first class. This is what a model that memorises titles can
    reach on ``title_seen``; NaN when no title is seen.
    """
    hits = total = 0
    for sample in samples:
        counts = labels.get(title_key(sample.text))
        if counts is None:
            continue
        best = max(sorted(counts), key=counts.__getitem__)
        hits += best == sample.label
        total += 1
    return hits / total if total else float("nan")


def generic_titles(labels: Mapping[str, Counter[str]], top: int = 10) -> list[dict[str, object]]:
    """Most frequent training titles, with the number of classes they span."""
    ranked = sorted(labels.items(), key=lambda kv: (-kv[1].total(), kv[0]))[:top]
    return [
        {
            "title": key,
            "count": counts.total(),
            "classes": len(counts),
            "top": counts.most_common(3),
        }
        for key, counts in ranked
    ]


def file_digest(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def image_duplicates(
    root: Path, train: Sequence[Sample], others: Sequence[Sample]
) -> list[dict[str, object]]:
    """Samples of ``others`` whose image file is byte-identical to a training image."""
    by_digest: dict[str, Sample] = {}
    for sample in train:
        by_digest.setdefault(file_digest(sample.image_path(root)), sample)
    found = []
    for sample in others:
        match = by_digest.get(file_digest(sample.image_path(root)))
        if match is not None:
            found.append(
                {
                    "id": sample.image_name,
                    "train_id": match.image_name,
                    "same_label": match.label == sample.label,
                }
            )
    return found


def image_subsets(
    samples: Sequence[Sample], members: Mapping[str, Sequence[str]], duplicated: set[str]
) -> dict[str, list[str]]:
    """``image_seen`` / ``image_unseen`` and ``unseen`` (neither title nor image seen)."""
    title_unseen = set(members["title_unseen"])
    ids = [s.image_name for s in samples]
    return {
        "image_seen": [i for i in ids if i in duplicated],
        "image_unseen": [i for i in ids if i not in duplicated],
        "unseen": [i for i in ids if i not in duplicated and i in title_unseen],
    }


def split_report(
    samples: Sequence[Sample],
    members: Mapping[str, Sequence[str]],
    labels: Mapping[str, Counter[str]],
) -> dict[str, object]:
    n = len(samples)
    seen = set(members["title_seen"])
    no_name = set(members["no_class_name"])
    return {
        "n": n,
        **{name: {"n": len(ids), "frac": len(ids) / n} for name, ids in members.items()},
        "seen_and_no_class_name": len(seen & no_name),
        "lookup_top1_on_title_seen": lookup_accuracy(samples, labels),
    }


def make_subsets(cfg: DictConfig) -> dict[str, object]:
    root = Path(cfg.paths.upmc_dir)
    train, val = split_train_val(
        read_split(root, "train"), read_ids(Path(to_absolute_path(cfg.val_ids)))
    )
    by_split = {"val": val, "test": read_split(root, "test")}
    labels = title_labels(train)
    out_dir = Path(to_absolute_path(cfg.out_dir))

    report: dict[str, object] = {
        "train_titles": {
            "n": len(train),
            "distinct": len(labels),
            "spanning_several_classes": sum(len(c) > 1 for c in labels.values()),
        },
        "generic_titles": generic_titles(labels),
    }
    for split, samples in by_split.items():
        members = subsets(samples, labels)
        if cfg.check_images:
            dups = image_duplicates(root, train, samples)
            members.update(image_subsets(samples, members, {str(d["id"]) for d in dups}))
            same_label = sum(bool(d["same_label"]) for d in dups)
            report[f"{split}_image_duplicates"] = {
                "n": len(dups),
                "same_label": same_label,
                "examples": dups[:20],
            }
            log.info(
                "%s: %d images identical to a training image, %d with the same class",
                split,
                len(dups),
                same_label,
            )
        for name, ids in members.items():
            write_ids(out_dir / f"{split}_{name}.txt", ids, overwrite=True)
        stats = split_report(samples, members, labels)
        report[split] = stats
        log.info("%s: %s", split, json.dumps(stats))

    report_path = Path(to_absolute_path(cfg.report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("subsets in %s, report in %s", out_dir, report_path)
    return report


@hydra.main(version_base=None, config_path="../../../configs", config_name="subsets")
def main(cfg: DictConfig) -> None:
    make_subsets(cfg)


if __name__ == "__main__":
    main()
