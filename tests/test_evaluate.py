from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from recipe_fusion.evaluate import (
    aggregate,
    align,
    evaluate,
    find_runs,
    macro_f1,
    mcnemar,
    metrics,
    run_evaluation,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    logits = np.zeros((len(labels), n_classes), dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 1.0
    return logits


def save_preds(path: Path, ids: list[str], logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), logits=logits)


def test_perfect_predictions() -> None:
    labels = np.array([0, 1, 2, 2])
    assert metrics(one_hot(labels, 3), labels) == {"top1": 1.0, "top5": 1.0, "macro_f1": 1.0}


def test_random_predictions_are_near_chance() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 101, size=20_000)
    result = metrics(rng.standard_normal((20_000, 101)), labels)
    assert abs(result["top1"] - 1 / 101) < 0.005
    assert abs(result["top5"] - 5 / 101) < 0.01


def test_macro_f1_matches_scikit_learn() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    labels, pred = rng.integers(0, 7, size=500), rng.integers(0, 7, size=500)
    expected = sklearn_metrics.f1_score(labels, pred, average="macro")
    assert macro_f1(pred, labels, 7) == pytest.approx(expected)


def test_top5_counts_the_five_best_scores() -> None:
    logits = np.array([[6, 5, 4, 3, 2, 1, 0]], dtype=float)
    assert metrics(logits, np.array([4]))["top5"] == 1.0
    assert metrics(logits, np.array([5]))["top5"] == 0.0
    assert metrics(logits, np.array([4]))["top1"] == 0.0


def test_align_reorders_by_id() -> None:
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    aligned = align(["b", "a"], logits, ["a", "b"], n_classes=2)
    np.testing.assert_array_equal(aligned, logits[::-1])


@pytest.mark.parametrize(
    ("ids", "logits", "match"),
    [
        (["a", "a"], np.zeros((2, 2)), "duplicate"),
        (["a"], np.zeros((1, 2)), "1 missing"),
        (["a", "b", "c"], np.zeros((3, 2)), "1 extra"),
        (["a", "b"], np.zeros((2, 3)), "shape"),
        (["a", "b"], np.full((2, 2), np.nan), "NaN"),
    ],
)
def test_align_rejects_bad_predictions(ids, logits, match) -> None:
    with pytest.raises(ValueError, match=match):
        align(ids, logits, ["a", "b"], n_classes=2)


def test_mcnemar() -> None:
    a = np.array([True] * 10 + [False] * 10)
    assert mcnemar(a, a)["p_value"] == 1.0
    result = mcnemar(np.array([True] * 10), np.array([False] * 10))
    assert result["a_only"] == 10 and result["b_only"] == 0
    assert result["p_value"] == pytest.approx(2 / 2**10)


def test_find_runs_and_aggregate_over_seeds(tmp_path: Path) -> None:
    ids, labels = ["a", "b", "c", "d"], np.array([0, 1, 0, 1])
    good, bad = one_hot(labels, 2), one_hot(1 - labels, 2)
    save_preds(tmp_path / "m" / "seed0" / "val_both.npz", ids, good)
    save_preds(tmp_path / "m" / "seed1" / "val_both.npz", ids, bad)
    save_preds(tmp_path / "m" / "seed0" / "test_both.npz", ids, good)
    save_preds(tmp_path / "m" / "notaseed" / "val_both.npz", ids, good)
    runs = find_runs(tmp_path, "val")
    assert [(r.method, r.seed, r.condition) for r in runs] == [("m", 0, "both"), ("m", 1, "both")]

    rows = evaluate(runs, ids, labels, {"first_two": ["a", "b"]}, n_classes=2)
    summary = {(r["subset"], r["metric"]): r for r in aggregate(rows)}
    top1 = summary[("all", "top1")]
    assert top1["mean"] == 0.5 and top1["seeds"] == 2
    assert top1["std"] == pytest.approx(np.std([1.0, 0.0], ddof=1))
    assert ("first_two", "top1") in summary


def test_run_evaluation_on_validation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F101_DATA_DIR", str(tmp_path))
    titles = tmp_path / "upmc-food101" / "texts" / "train_titles.csv"
    titles.parent.mkdir(parents=True)
    names = [f"{label}_{i}.jpg" for label in ("a", "b") for i in range(5)]
    titles.write_text("".join(f"{n},T,{n[0]}\n" for n in names), encoding="utf-8")
    val_ids = tmp_path / "val_ids.txt"
    val_ids.write_text("a_0.jpg\nb_0.jpg\n", encoding="utf-8")
    preds = tmp_path / "preds"
    save_preds(preds / "m" / "seed0" / "val_both.npz", ["b_0.jpg", "a_0.jpg"], np.eye(2)[::-1])

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(
            config_name="evaluate",
            overrides=[
                f"val_ids={val_ids}",
                f"preds_dir={preds}",
                f"results_dir={tmp_path / 'results'}",
                "subsets=null",
            ],
        )
    summary = run_evaluation(cfg)
    assert {(r["metric"], r["mean"]) for r in summary if r["subset"] == "all"} >= {("top1", 1.0)}
    assert (tmp_path / "results" / "val.csv").is_file()
    assert (tmp_path / "results" / "val_summary.csv").is_file()
