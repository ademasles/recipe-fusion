import numpy as np
import pytest
import torch
import torch.nn.functional as F

from recipe_fusion.seed import seed_everything
from recipe_fusion.training import MLPHead, fit, load_head, predict_logits, save_head

CPU = torch.device("cpu")


def blobs(n_per_class: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    x = rng.standard_normal((len(y), 8)).astype(np.float32)
    x[np.arange(len(y)), y] += 4.0
    order = rng.permutation(len(y))
    return x[order].astype(np.float16), y[order]


def new_head() -> MLPHead:
    return MLPHead(dim_in=8, hidden=16, n_classes=3, dropout=0.1)


def run_fit(model: MLPHead, y_val: np.ndarray | None = None, **kwargs):
    x_train, y_train = blobs(seed=0)
    x_val, y_val_true = blobs(n_per_class=20, seed=1)
    options = {"lr": 1e-2, "weight_decay": 1e-4, "seed": 0, "device": CPU, "batch_size": 32}
    options.update(kwargs)
    labels = y_val_true if y_val is None else y_val
    return fit(model, x_train, y_train, x_val, labels, **options)


def test_head_output_shape() -> None:
    assert new_head()(torch.zeros(5, 8)).shape == (5, 3)


def test_fit_learns_and_restores_best_weights() -> None:
    seed_everything(0)
    model = new_head()
    result = run_fit(model, max_epochs=20, patience=3)
    assert result.val_top1 > 0.95
    assert 1 <= result.best_epoch <= result.epochs_run <= 20
    x_val, y_val = blobs(n_per_class=20, seed=1)
    logits = torch.from_numpy(predict_logits(model, x_val, device=CPU))
    loss = F.cross_entropy(logits, torch.as_tensor(y_val)).item()
    assert loss == pytest.approx(result.val_loss, abs=1e-5)


def test_early_stopping_waits_for_patience() -> None:
    noise = np.random.default_rng(2).integers(0, 3, size=60)  # unlearnable validation labels
    seed_everything(0)
    result = run_fit(new_head(), y_val=noise, max_epochs=50, patience=2)
    assert result.epochs_run == result.best_epoch + 2 < 50


def test_fit_is_deterministic_for_a_seed() -> None:
    x_val, _ = blobs(n_per_class=20, seed=1)
    outputs = []
    for _ in range(2):
        seed_everything(0)
        model = new_head()
        run_fit(model, max_epochs=3)
        outputs.append(predict_logits(model, x_val, device=CPU))
    np.testing.assert_allclose(outputs[0], outputs[1])


def test_fit_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="max_epochs"):
        run_fit(new_head(), max_epochs=0)


def test_head_checkpoint_round_trip(tmp_path) -> None:
    seed_everything(0)
    model = new_head()
    x, _ = blobs()
    save_head(tmp_path / "ckpt" / "seed0.pt", model, lr=0.01, seed=0)
    loaded = load_head(tmp_path / "ckpt" / "seed0.pt")
    np.testing.assert_allclose(
        predict_logits(loaded, x, device=CPU), predict_logits(model, x, device=CPU)
    )
