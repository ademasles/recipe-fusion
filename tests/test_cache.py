from pathlib import Path

import numpy as np
import pytest

from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import labels_for, read_features, write_features

IDS = ["a_1.jpg", "a_2.jpg", "b_1.jpg"]


def arrays(n: int = 3, length: int = 4, dim: int = 8):
    rng = np.random.default_rng(0)
    pooled = rng.standard_normal((n, dim)).astype(np.float16)
    tokens = rng.standard_normal((n, length, dim)).astype(np.float16)
    mask = np.ones((n, length), dtype=bool)
    mask[:, -1] = False
    return pooled, tokens, mask


def test_round_trip_with_tokens(tmp_path: Path) -> None:
    pooled, tokens, mask = arrays()
    write_features(tmp_path, "train", IDS, pooled, tokens, mask, meta={"encoder": "x"})
    features = read_features(tmp_path, "train")
    assert features.ids == IDS
    assert isinstance(features.pooled, np.memmap)
    assert features.tokens is not None and features.mask is not None
    np.testing.assert_array_equal(features.pooled, pooled)
    np.testing.assert_array_equal(features.tokens, tokens)
    np.testing.assert_array_equal(features.mask, mask)
    assert features.meta == {"encoder": "x"}


def test_round_trip_pooled_only_without_mmap(tmp_path: Path) -> None:
    pooled, _, _ = arrays()
    write_features(tmp_path, "test", IDS, pooled)
    features = read_features(tmp_path, "test", mmap=False)
    assert not isinstance(features.pooled, np.memmap)
    assert features.tokens is None and features.mask is None


def test_meta_is_kept_per_split(tmp_path: Path) -> None:
    pooled, _, _ = arrays()
    write_features(tmp_path, "train", IDS, pooled, meta={"seconds": 10})
    write_features(tmp_path, "test", IDS, pooled, meta={"seconds": 3})
    assert read_features(tmp_path, "train").meta == {"seconds": 10}
    assert read_features(tmp_path, "test").meta == {"seconds": 3}


def test_rewrite_without_tokens_removes_stale_token_files(tmp_path: Path) -> None:
    pooled, tokens, mask = arrays()
    write_features(tmp_path, "train", IDS, pooled, tokens, mask)
    write_features(tmp_path, "train", IDS, pooled)
    assert read_features(tmp_path, "train").tokens is None


def test_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_features(tmp_path, "train")


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda p, t, m: (["a"] * 3, p, t, m), ValueError, "duplicate"),
        (lambda p, t, m: (IDS, p.astype(np.float32), t, m), TypeError, "pooled"),
        (lambda p, t, m: (IDS, p[:2], t, m), ValueError, "pooled"),
        (lambda p, t, m: (IDS, p, t, None), ValueError, "together"),
        (lambda p, t, m: (IDS, p, t.astype(np.float32), m), TypeError, "tokens"),
        (lambda p, t, m: (IDS, p, t, m.astype(np.int8)), TypeError, "mask"),
        (lambda p, t, m: (IDS, p, t[:2], m[:2]), ValueError, "tokens"),
        (lambda p, t, m: (IDS, p, t, m[:, :2]), ValueError, "mask"),
        (lambda p, t, m: (IDS, p, t, np.zeros_like(m)), ValueError, "no real token"),
        (lambda p, t, m: (IDS, np.full_like(p, np.nan), t, m), ValueError, "NaN"),
        (lambda p, t, m: (IDS, p, np.full_like(t, np.inf), m), ValueError, "NaN"),
    ],
)
def test_invalid_features_are_rejected(tmp_path: Path, change, error, match) -> None:
    ids, pooled, tokens, mask = change(*arrays())
    with pytest.raises(error, match=match):
        write_features(tmp_path, "train", ids, pooled, tokens, mask)
    assert not (tmp_path / "train_ids.txt").exists()


def test_labels_for_follows_ids_order() -> None:
    samples = [Sample(name, "", name[0], "train") for name in IDS]
    labels = labels_for(["b_1.jpg", "a_1.jpg"], samples, ["a", "b"])
    np.testing.assert_array_equal(labels, [1, 0])
    with pytest.raises(KeyError, match=r"c_1\.jpg"):
        labels_for(["c_1.jpg"], samples, ["a", "b"])


def test_tokens_already_memmapped_at_destination_are_not_rewritten(tmp_path: Path) -> None:
    pooled, tokens, mask = arrays()
    path = tmp_path / "train_tokens.npy"
    on_disk = np.lib.format.open_memmap(path, mode="w+", dtype=tokens.dtype, shape=tokens.shape)
    on_disk[:] = tokens
    on_disk.flush()
    modified = path.stat().st_mtime_ns

    write_features(tmp_path, "train", IDS, pooled, on_disk, mask)
    assert path.stat().st_mtime_ns == modified
    features = read_features(tmp_path, "train", mmap=False)
    assert features.tokens is not None
    np.testing.assert_array_equal(features.tokens, tokens)


def test_non_finite_value_in_last_chunk_is_rejected(tmp_path: Path) -> None:
    # 1_000_000 values per chunk: 2 rows of 600_000 put the last row in a second chunk.
    pooled = np.zeros((2, 600_000), dtype=np.float16)
    pooled[-1, -1] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        write_features(tmp_path, "train", ["a.jpg", "b.jpg"], pooled)
