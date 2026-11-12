"""Text reference without a network: TF-IDF + logistic regression (sheet B2 a).

Method ``tfidf_logreg``. Fitted on the training set without the validation subset; C is chosen
on the validation top-1 (ties keep the smaller C). The fit is deterministic, so the same logits
are written for every seed. Test predictions are written for the chosen C only.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from recipe_fusion.data.clean import clean_title
from recipe_fusion.data.loading import Splits, load_splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import labels_for
from recipe_fusion.preds import preds_path, write_preds

log = logging.getLogger(__name__)


def titles(samples: Sequence[Sample]) -> list[str]:
    """The shared cleaning (same text as the BERT models), then lowercase."""
    return [clean_title(s.text).lower() for s in samples]


def names(samples: Sequence[Sample]) -> list[str]:
    return [s.image_name for s in samples]


def top1(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def run(cfg: DictConfig, splits: Splits) -> dict[str, Any]:
    outputs = Path(to_absolute_path(cfg.outputs_dir))
    method, condition = str(cfg.method), str(cfg.condition)
    n_classes = len(splits.classes)
    y_train = labels_for(names(splits.train), splits.train_all, splits.classes)
    y_val = labels_for(names(splits.val), splits.train_all, splits.classes)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    x_train = vectorizer.fit_transform(titles(splits.train))
    x_val = vectorizer.transform(titles(splits.val))
    x_test = vectorizer.transform(titles(splits.test))

    trials: list[dict[str, float]] = []
    best: tuple[dict[str, float], LogisticRegression] | None = None
    for c in cfg.c_grid:
        clf = LogisticRegression(C=float(c), max_iter=int(cfg.max_iter)).fit(x_train, y_train)
        if not np.array_equal(clf.classes_, np.arange(n_classes)):
            raise ValueError("some classes are missing from the training part of the split")
        trial = {"C": float(c), "val_top1": top1(clf.decision_function(x_val), y_val)}
        trials.append(trial)
        log.info("C=%g -> val top-1 %.4f", c, trial["val_top1"])
        if best is None or trial["val_top1"] > best[0]["val_top1"]:
            best = (trial, clf)
    if best is None:
        raise ValueError("c_grid is empty")
    chosen, clf = best

    val_logits = clf.decision_function(x_val).astype(np.float32)
    test_logits = clf.decision_function(x_test).astype(np.float32)
    seeds = [int(s) for s in cfg.seeds]
    for seed in seeds:
        write_preds(
            preds_path(outputs, method, seed, "val", condition),
            names(splits.val),
            val_logits,
            n_classes=n_classes,
        )
        write_preds(
            preds_path(outputs, method, seed, "test", condition),
            names(splits.test),
            test_logits,
            n_classes=n_classes,
        )
    return {"trials": trials, "chosen": chosen, "seeds": seeds}


@hydra.main(version_base=None, config_path="../../../configs", config_name="train/tfidf_logreg")
def main(cfg: DictConfig) -> None:
    splits = load_splits(
        Path(to_absolute_path(cfg.paths.upmc_dir)), Path(to_absolute_path(cfg.val_ids))
    )
    summary = run(cfg, splits)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("chosen C=%g, val top-1 %.4f", summary["chosen"]["C"], summary["chosen"]["val_top1"])


if __name__ == "__main__":
    main()
