import numpy as np
import pytest

from recipe_fusion.preds import ckpt_path, preds_path, write_preds

IDS = ["a_1.jpg", "a_2.jpg", "b_1.jpg"]


def test_paths_follow_the_contract(tmp_path) -> None:
    assert preds_path(tmp_path, "text_mlp", 1, "val", "text_only") == (
        tmp_path / "preds" / "text_mlp" / "seed1" / "val_text_only.npz"
    )
    assert ckpt_path(tmp_path, "text_mlp", 1) == tmp_path / "ckpt" / "text_mlp" / "seed1.pt"


def test_write_preds_round_trip(tmp_path) -> None:
    logits = np.random.default_rng(0).standard_normal((3, 101)).astype(np.float32)
    path = preds_path(tmp_path, "m", 0, "val", "both")
    write_preds(path, IDS, logits)
    with np.load(path) as data:
        assert data["ids"].tolist() == IDS
        assert data["logits"].dtype == np.float32
        np.testing.assert_array_equal(data["logits"], logits)


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda i, z: (["a"] * 3, z), ValueError, "duplicate"),
        (lambda i, z: (i, z.astype(np.float64)), TypeError, "float32"),
        (lambda i, z: (i, z[:2]), ValueError, "shape"),
        (lambda i, z: (i, z[:, :100]), ValueError, "shape"),
        (lambda i, z: (i, np.full_like(z, np.nan)), ValueError, "NaN"),
    ],
)
def test_invalid_preds_are_rejected(tmp_path, change, error, match) -> None:
    ids, logits = change(IDS, np.zeros((3, 101), dtype=np.float32))
    with pytest.raises(error, match=match):
        write_preds(tmp_path / "p.npz", ids, logits)
    assert not (tmp_path / "p.npz").exists()
