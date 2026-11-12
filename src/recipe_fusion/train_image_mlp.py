"""Train image_mlp heads on cached, frozen ViT features."""

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from recipe_fusion.data.split import read_ids
from recipe_fusion.data.upmc import class_names, read_split
from recipe_fusion.features.cache import Features, read_features, validate
from recipe_fusion.models.image_mlp import FEATURE_DIM, NUM_CLASSES, ImageMLP
from recipe_fusion.seed import pick_device, seed_everything

log = logging.getLogger(__name__)

ENCODER_NAME = "vit_b_16_swag"
EXPECTED_PREPROCESSING = "weights.transforms()"
EXPECTED_POOLING = "CLS token after the final LayerNorm"
REFERENCE_TOP1 = 0.7283
VALIDATION_GAP_WARNING = 0.10


def _validate_cache(
    features: Features,
    samples,
    split: str,
    *,
    expected_encoder: str,
) -> None:
    """Check cache structure, preprocessing metadata, and exact annotation alignment."""
    validate(features)
    if features.pooled.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{split} image features must have dimension {FEATURE_DIM}, "
            f"got {features.pooled.shape[1]}"
        )
    expected_ids = [sample.image_name for sample in samples]
    if features.ids != expected_ids:
        actual, expected = set(features.ids), set(expected_ids)
        raise ValueError(
            f"{split} feature ids do not exactly match {split}_titles.csv "
            f"(missing={len(expected - actual)}, extra={len(actual - expected)}, "
            f"order_matches={features.ids == expected_ids}); re-extract features"
        )
    metadata_checks = {
        "encoder": expected_encoder,
        "preprocessing": EXPECTED_PREPROCESSING,
        "pooled": EXPECTED_POOLING,
    }
    for field, expected in metadata_checks.items():
        actual = features.meta.get(field)
        if actual != expected:
            raise ValueError(
                f"{split} feature metadata {field!r} must be {expected!r}, got {actual!r}; "
                "verify image preprocessing and encoder before training"
            )
    if features.meta.get("d") != FEATURE_DIM:
        raise ValueError(
            f"{split} feature metadata 'd' must be {FEATURE_DIM}, got {features.meta.get('d')!r}"
        )


def _labels(ids: list[str], samples, classes: list[str], split: str) -> npt.NDArray[np.int64]:
    """Map each cached image id to the label in the source title file."""
    label_by_id = {sample.image_name: sample.label for sample in samples}
    if len(label_by_id) != len(samples):
        raise ValueError(f"{split} annotations contain duplicate image ids")
    class_index = {name: index for index, name in enumerate(classes)}
    unknown = [image_id for image_id in ids if image_id not in label_by_id]
    if unknown:
        raise ValueError(f"{split} ids have no annotation, e.g. {unknown[0]!r}")
    unknown_labels = sorted({label_by_id[image_id] for image_id in ids} - set(class_index))
    if unknown_labels:
        raise ValueError(f"{split} annotations contain unknown classes: {unknown_labels}")
    return np.asarray([class_index[label_by_id[image_id]] for image_id in ids], dtype=np.int64)


def _features_tensor(features: Features, indices: npt.NDArray[np.int64]) -> Tensor:
    values = np.asarray(features.pooled[indices], dtype=np.float32)
    return torch.from_numpy(values)


def _logits(
    model: nn.Module,
    features: Features,
    indices: npt.NDArray[np.int64],
    device: torch.device,
    batch_size: int,
) -> npt.NDArray[np.float32]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = _features_tensor(features, indices[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def _save_predictions(path: Path, ids: list[str], logits: npt.NDArray[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), logits=logits)


def _warn_if_validation_low(seed: int, top1: float) -> None:
    """Flag a large validation gap; this external result is only a rough sanity reference."""
    threshold = REFERENCE_TOP1 - VALIDATION_GAP_WARNING
    if top1 < threshold:
        log.warning(
            "image_mlp seed %d validation top-1 is %.2f%% (rough warning threshold %.2f%%, "
            "based on 72.83%% minus 10 points). Before changing hyperparameters, inspect "
            "feature preprocessing and image-id/label alignment. The reference used test "
            "selection and augmentation, so this is not a directly comparable target.",
            seed,
            top1 * 100,
            threshold * 100,
        )


def train_seed(
    *,
    seed: int,
    train_features: Features,
    train_labels: npt.NDArray[np.int64],
    train_indices: npt.NDArray[np.int64],
    val_indices: npt.NDArray[np.int64],
    val_ids: list[str],
    val_labels: npt.NDArray[np.int64],
    test_features: Features,
    classes: list[str],
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    checkpoint_path: Path,
    predictions_dir: Path,
) -> dict[str, Any]:
    """Train one seed and select its checkpoint using validation top-1, then NLL."""
    seed_everything(seed)
    model = ImageMLP(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_x = _features_tensor(train_features, train_indices)
    train_y = torch.from_numpy(train_labels[train_indices])
    val_x = _features_tensor(train_features, val_indices)
    val_y = torch.from_numpy(val_labels)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_top1 = -1.0
    best_nll = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for features_batch, labels_batch in tqdm(
            loader, desc=f"seed {seed} epoch {epoch}/{epochs}", leave=False
        ):
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features_batch), labels_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_logits_tensor = model(val_x.to(device))
            val_nll = float(criterion(val_logits_tensor, val_y.to(device)).item())
            val_top1 = float((val_logits_tensor.argmax(dim=1).cpu() == val_y).float().mean().item())
        log.info(
            "seed %d epoch %d: validation top-1 %.4f, NLL %.6f",
            seed,
            epoch,
            val_top1,
            val_nll,
        )

        improved = val_top1 > best_top1 or (val_top1 == best_top1 and val_nll < best_nll)
        if improved:
            best_top1, best_nll, best_epoch = val_top1, val_nll, epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_top1": best_top1,
                    "best_val_nll": best_nll,
                    "encoder": ENCODER_NAME,
                    "preprocessing": EXPECTED_PREPROCESSING,
                    "pooled": EXPECTED_POOLING,
                    "architecture": (
                        "LayerNorm(768)-Linear(768,512)-GELU-Dropout(0.3)-Linear(512,101)"
                    ),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("seed %d: early stopping after epoch %d", seed, epoch)
                break

    if best_epoch == 0:
        raise RuntimeError(f"seed {seed} did not produce a valid checkpoint")
    _warn_if_validation_low(seed, best_top1)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_logits = _logits(model, train_features, val_indices, device, batch_size)
    test_indices = np.arange(len(test_features), dtype=np.int64)
    test_logits = _logits(model, test_features, test_indices, device, batch_size)
    _save_predictions(predictions_dir / f"seed{seed}" / "val_image.npz", val_ids, val_logits)
    _save_predictions(
        predictions_dir / f"seed{seed}" / "test_image.npz", test_features.ids, test_logits
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "best_val_nll": best_nll,
    }


def run_training(cfg: DictConfig) -> list[dict[str, Any]]:
    """Preflight all data, then train configured seeds and save checkpoints and logits."""
    data_root = Path(to_absolute_path(cfg.paths.upmc_dir))
    cache_dir = Path(to_absolute_path(cfg.paths.features_dir)) / "image" / cfg.encoder
    train_samples = read_split(data_root, "train")
    test_samples = read_split(data_root, "test")
    classes = class_names(train_samples)
    if len(classes) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} training classes, found {len(classes)}")

    train_features = read_features(cache_dir, "train")
    test_features = read_features(cache_dir, "test")
    _validate_cache(train_features, train_samples, "train", expected_encoder=str(cfg.encoder))
    _validate_cache(test_features, test_samples, "test", expected_encoder=str(cfg.encoder))
    if set(train_features.ids) & set(test_features.ids):
        raise ValueError("training and test feature ids overlap")

    val_ids = read_ids(Path(to_absolute_path(cfg.val_ids)))
    if not val_ids or len(set(val_ids)) != len(val_ids):
        raise ValueError("validation ids must be non-empty and unique")
    train_id_to_index = {image_id: index for index, image_id in enumerate(train_features.ids)}
    unknown_val_ids = [image_id for image_id in val_ids if image_id not in train_id_to_index]
    if unknown_val_ids:
        raise ValueError(
            f"{len(unknown_val_ids)} validation ids are not in the training cache, "
            f"e.g. {unknown_val_ids[0]!r}"
        )

    val_set = set(val_ids)
    val_indices = np.asarray([train_id_to_index[image_id] for image_id in val_ids], dtype=np.int64)
    train_indices = np.asarray(
        [i for i, image_id in enumerate(train_features.ids) if image_id not in val_set],
        dtype=np.int64,
    )
    if not len(train_indices):
        raise ValueError("no training examples remain after applying the validation split")
    train_labels = _labels(train_features.ids, train_samples, classes, "train")
    val_labels = _labels(val_ids, train_samples, classes, "validation")
    for name, labels in (("train", train_labels[train_indices]), ("validation", val_labels)):
        if set(np.unique(labels)) != set(range(NUM_CLASSES)):
            raise ValueError(f"{name} split does not contain all {NUM_CLASSES} classes")

    batch_size = int(cfg.train.batch_size)
    epochs = int(cfg.train.epochs)
    patience = int(cfg.train.patience)
    learning_rate = float(cfg.train.learning_rate)
    weight_decay = float(cfg.train.weight_decay)
    if batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("batch_size, epochs, and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    seeds = [int(seed) for seed in cfg.train.seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")

    log.info(
        "preflight passed: encoder=%s, preprocessing=%s, pooled=%s, "
        "train=%d, validation=%d, test=%d, classes=%d; ids and labels aligned",
        cfg.encoder,
        EXPECTED_PREPROCESSING,
        EXPECTED_POOLING,
        len(train_indices),
        len(val_ids),
        len(test_features),
        len(classes),
    )
    log.info(
        "validation sanity reference: warn below %.2f%% top-1; this is approximate, "
        "not a target or a hyperparameter-selection rule",
        (REFERENCE_TOP1 - VALIDATION_GAP_WARNING) * 100,
    )

    device = pick_device(str(cfg.train.device))
    checkpoint_dir = Path(to_absolute_path(cfg.checkpoint_dir))
    predictions_dir = Path(to_absolute_path(cfg.preds_dir)) / "image_mlp"
    results = []
    for seed in seeds:
        result = train_seed(
            seed=seed,
            train_features=train_features,
            train_labels=train_labels,
            train_indices=train_indices,
            val_indices=val_indices,
            val_ids=val_ids,
            val_labels=val_labels,
            test_features=test_features,
            classes=classes,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_dir / f"seed{seed}.pt",
            predictions_dir=predictions_dir,
        )
        results.append(result)
        log.info(
            "image_mlp seed %d: best epoch %d, validation top-1 %.4f",
            result["seed"],
            result["best_epoch"],
            result["best_val_top1"],
        )
    return results


@hydra.main(version_base=None, config_path="../../configs", config_name="image_mlp")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
