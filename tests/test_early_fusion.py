import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from recipe_fusion.data.loading import Splits
from recipe_fusion.data.upmc import Sample
from recipe_fusion.features.cache import write_features
from recipe_fusion.models import early_fusion
from recipe_fusion.models.heads import EarlyConcatHead, load_early_concat, save_head
from recipe_fusion.preds import ckpt_path, preds_path


def vectors(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n, 4)).astype(np.float16),
        rng.standard_normal((n, 3)).astype(np.float16),
    )


def test_early_concat_head_normalizes_each_modality() -> None:
    torch.manual_seed(0)
    model = EarlyConcatHead(dim_text=4, dim_image=3, hidden=8, n_classes=2).eval()
    text, image = torch.randn(5, 4), torch.randn(5, 3)
    assert model(text=text, image=image).shape == (5, 2)
    # a much larger image scale must not change the output: the image has its own LayerNorm
    torch.testing.assert_close(
        model(text=text, image=image), model(text=text, image=image * 1000), atol=1e-3, rtol=0
    )


def test_early_concat_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    model = EarlyConcatHead(dim_text=4, dim_image=3, hidden=8, n_classes=2).eval()
    save_head(tmp_path / "c.pt", model)
    loaded = load_early_concat(tmp_path / "c.pt").eval()
    text, image = torch.randn(5, 4), torch.randn(5, 3)
    torch.testing.assert_close(loaded(text=text, image=image), model(text=text, image=image))


def test_pair_batches_serve_both_modalities() -> None:
    text, image = vectors()
    batches = early_fusion.PairBatches(text, image)
    got = batches.get(np.array([1, 3]))
    assert len(batches) == 10 and set(got) == {"text", "image"}
    np.testing.assert_array_equal(got["text"].numpy(), text[[1, 3]])
    np.testing.assert_array_equal(got["image"].numpy(), image[[1, 3]])
    with pytest.raises(ValueError, match="rows"):
        early_fusion.PairBatches(text, image[:5])


def test_ablate_conditions() -> None:
    text, image = vectors()
    kwargs = {
        "mean_text": text.astype(np.float32).mean(axis=0),
        "mean_image": image.astype(np.float32).mean(axis=0),
        "seed": 0,
    }
    both = early_fusion.ablate(text, image, "both", **kwargs)
    np.testing.assert_array_equal(both[0], text)
    np.testing.assert_array_equal(both[1], image)

    text_c, image_c = early_fusion.ablate(text, image, "text_mean", **kwargs)
    np.testing.assert_allclose(text_c, np.tile(kwargs["mean_text"], (10, 1)))
    np.testing.assert_array_equal(image_c, image)

    text_c, image_c = early_fusion.ablate(text, image, "image_mean", **kwargs)
    np.testing.assert_allclose(image_c, np.tile(kwargs["mean_image"], (10, 1)))
    np.testing.assert_array_equal(text_c, text)

    text_c, image_c = early_fusion.ablate(text, image, "text_shuffled", **kwargs)
    np.testing.assert_array_equal(image_c, image)
    assert not np.array_equal(text_c, text)
    np.testing.assert_array_equal(np.sort(text_c, axis=0), np.sort(text, axis=0))
    again, _ = early_fusion.ablate(text, image, "text_shuffled", **kwargs)
    np.testing.assert_array_equal(text_c, again)  # fixed seed

    text_c, image_c = early_fusion.ablate(text, image, "image_shuffled", **kwargs)
    np.testing.assert_array_equal(text_c, text)
    np.testing.assert_array_equal(np.sort(image_c, axis=0), np.sort(image, axis=0))


def test_ablate_rejects_unknown_condition() -> None:
    text, image = vectors()
    with pytest.raises(ValueError, match="unknown condition"):
        early_fusion.ablate(text, image, "nope", mean_text=text[0], mean_image=image[0], seed=0)


def test_load_fusion_data_aligns_both_modalities_by_id(tmp_path) -> None:
    names = [f"{c}_{i}.jpg" for c in ("a", "b") for i in range(4)]
    train_all = [Sample(n, "t", n[0], "train") for n in names]
    test = [Sample("a_9.jpg", "t", "a", "test")]
    splits = Splits(train_all, train_all[1:], train_all[:1], test, ["a", "b"])
    text = np.arange(32, dtype=np.float16).reshape(8, 4)
    image = (100 + np.arange(24)).astype(np.float16).reshape(8, 3)
    for directory, array in ((tmp_path / "text", text), (tmp_path / "image", image)):
        write_features(directory, "train", names, array)
        write_features(directory, "test", ["a_9.jpg"], array[:1])

    data = early_fusion.load_fusion_data(splits, tmp_path / "text", tmp_path / "image")
    assert data.ids_val == [names[0]] and data.ids_test == ["a_9.jpg"]
    np.testing.assert_array_equal(data.text_val, text[:1])
    np.testing.assert_array_equal(data.image_train, image[1:])
    assert data.y_train.tolist() == [0, 0, 0, 1, 1, 1, 1]


def synthetic_data() -> early_fusion.FusionData:
    rng = np.random.default_rng(0)

    def block(n_per_class: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.repeat(np.arange(3), n_per_class)
        text = rng.standard_normal((len(y), 8)).astype(np.float32)
        text[np.arange(len(y)), y] += 4.0  # the class signal is only in the text
        image = rng.standard_normal((len(y), 6)).astype(np.float32)  # image = pure noise
        return text.astype(np.float16), image.astype(np.float16), y

    text_train, image_train, y_train = block(60)
    text_val, image_val, y_val = block(20)
    text_test, image_test, _ = block(10)
    return early_fusion.FusionData(
        classes=["a", "b", "c"],
        text_train=text_train,
        image_train=image_train,
        y_train=y_train,
        text_val=text_val,
        image_val=image_val,
        y_val=y_val,
        ids_val=[f"v{i}.jpg" for i in range(len(y_val))],
        text_test=text_test,
        image_test=image_test,
        ids_test=[f"t{i}.jpg" for i in range(len(text_test))],
    )


def make_cfg(tmp_path, **overrides) -> DictConfig:
    cfg = {
        "method": "early_concat",
        "outputs_dir": str(tmp_path),
        "seeds": [0, 1],
        "ablation_seed": 0,
        "lr": None,
        "weight_decay": None,
        "sweep": {"seed": 0, "lr": [0.01, 0.02], "weight_decay": [0.0001, 0.01]},
        "head": {"hidden": 16, "dropout": 0.1},
        "train": {"batch_size": 32, "max_epochs": 20, "patience": 3},
    }
    cfg.update(overrides)
    return DictConfig(cfg)


def test_run_writes_every_condition_and_ablations_behave(tmp_path) -> None:
    cfg = make_cfg(tmp_path)
    data = synthetic_data()
    summary = early_fusion.run(cfg, data, torch.device("cpu"))

    assert len(summary["trials"]) == 4
    sizes = {"val": len(data.ids_val), "test": len(data.ids_test)}
    for seed in (0, 1):
        for split in ("val", "test"):
            for condition in early_fusion.CONDITIONS:
                path = preds_path(tmp_path, "early_concat", seed, split, condition)
                with np.load(path) as saved:
                    assert saved["logits"].shape == (sizes[split], 3)
        assert load_early_concat(ckpt_path(tmp_path, "early_concat", seed)).args["n_classes"] == 3

    top1 = {c: stats["mean"] for c, stats in summary["val_top1"].items()}
    assert top1["both"] > 0.9
    assert top1["text_mean"] < top1["both"] - 0.2  # removing the informative modality hurts
    assert top1["image_mean"] > top1["text_mean"]
    assert (tmp_path / "preds" / "early_concat" / "meta.json").is_file()


def test_run_skips_the_search_when_lr_and_weight_decay_are_given(tmp_path) -> None:
    cfg = make_cfg(tmp_path, lr=0.01, weight_decay=0.0001, seeds=[0])
    summary = early_fusion.run(cfg, synthetic_data(), torch.device("cpu"))
    assert summary["trials"] == []
    assert summary["chosen"] == {"lr": 0.01, "weight_decay": 0.0001}
    assert preds_path(tmp_path, "early_concat", 0, "test", "both").is_file()
