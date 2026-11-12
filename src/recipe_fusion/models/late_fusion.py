"""Late fusion of unimodal prediction files; no model is trained here."""

import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from recipe_fusion.data.split import read_ids
from recipe_fusion.data.upmc import class_names, read_split
from recipe_fusion.evaluate import align
from recipe_fusion.preds import preds_path, write_preds

log = logging.getLogger(__name__)

CONDITIONS = ("both", "text_mean", "image_mean", "text_shuffled", "image_shuffled")
TEXT_ABLATIONS = frozenset(("text_mean", "text_shuffled"))
IMAGE_ABLATIONS = frozenset(("image_mean", "image_shuffled"))
TEMPERATURE_LOG_MIN = -2.0
TEMPERATURE_LOG_MAX = 2.0
TEMPERATURE_LOG_STEP = 0.1
WEIGHT_STEP = 0.05


def _log_softmax(logits: np.ndarray) -> npt.NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.ndim != 2 or len(logits) != len(labels) or not len(labels):
        raise ValueError("logits and labels must have compatible non-empty shapes")
    if labels.min() < 0 or labels.max() >= logits.shape[1]:
        raise ValueError("labels are outside the logit class range")
    log_probs = _log_softmax(logits)
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def calibrate_temperature(
    logits: np.ndarray, labels: np.ndarray, log_temperatures: np.ndarray | None = None
) -> tuple[float, float]:
    """Choose exp(log T) minimizing validation NLL on a grid from -2 to 2."""
    if log_temperatures is None:
        log_temperatures = np.linspace(
            TEMPERATURE_LOG_MIN,
            TEMPERATURE_LOG_MAX,
            round((TEMPERATURE_LOG_MAX - TEMPERATURE_LOG_MIN) / TEMPERATURE_LOG_STEP) + 1,
        )
    if log_temperatures.ndim != 1 or not len(log_temperatures):
        raise ValueError("log temperature grid must be a non-empty vector")
    candidates = []
    for log_temperature in log_temperatures:
        temperature = math.exp(float(log_temperature))
        loss = negative_log_likelihood(logits / temperature, labels)
        candidates.append((loss, abs(float(log_temperature)), float(log_temperature), temperature))
    loss, _, _, temperature = min(candidates)
    return temperature, loss


def calibrated_log_probs(logits: np.ndarray, temperature: float) -> npt.NDArray[np.float64]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    return _log_softmax(logits / temperature)


def choose_weight(
    image_logits: np.ndarray,
    text_logits: np.ndarray,
    labels: np.ndarray,
    image_temperature: float,
    text_temperature: float,
) -> tuple[float, float, float]:
    """Maximize validation top-1 over w; break ties by fused validation NLL."""
    image_log_probs = calibrated_log_probs(image_logits, image_temperature)
    text_log_probs = calibrated_log_probs(text_logits, text_temperature)
    candidates = []
    for step in range(round(1 / WEIGHT_STEP) + 1):
        weight = step * WEIGHT_STEP
        fused = weight * image_log_probs + (1 - weight) * text_log_probs
        top1 = float((fused.argmax(axis=1) == labels).mean())
        nll = negative_log_likelihood(fused, labels)
        candidates.append((-top1, nll, weight, top1))
    _, nll, weight, top1 = min(candidates)
    return weight, top1, nll


def _read_logits(path: Path, expected_ids: list[str], n_classes: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing required prediction file: {path}. Ablations must be recalculated by "
            "the corresponding unimodal model and saved in the shared prediction format."
        )
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) < {"ids", "logits"}:
            raise ValueError(f"{path}: expected arrays 'ids' and 'logits'")
        ids = [str(image_id) for image_id in data["ids"]]
        logits = np.asarray(data["logits"])
    return align(ids, logits, expected_ids, n_classes)


def _labels_and_ids(
    upmc_dir: Path, val_ids_path: Path
) -> tuple[list[str], npt.NDArray[np.int64], list[str], list[str]]:
    train_samples = read_split(upmc_dir, "train")
    test_samples = read_split(upmc_dir, "test")
    classes = class_names(train_samples)
    val_ids = read_ids(val_ids_path)
    if not val_ids or len(set(val_ids)) != len(val_ids):
        raise ValueError("validation ids must be non-empty and unique")

    sample_by_id = {sample.image_name: sample for sample in train_samples}
    if len(sample_by_id) != len(train_samples):
        raise ValueError("training annotations contain duplicate image ids")
    unknown_val_ids = [image_id for image_id in val_ids if image_id not in sample_by_id]
    if unknown_val_ids:
        raise ValueError(
            f"validation ids are not in train annotations, e.g. {unknown_val_ids[0]!r}"
        )
    val_set = set(val_ids)
    train_val_ids = {sample.image_name for sample in train_samples}
    if val_set == train_val_ids:
        raise ValueError("validation ids contain every training sample")
    test_ids = [sample.image_name for sample in test_samples]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("test annotations contain duplicate image ids")

    class_index = {name: index for index, name in enumerate(classes)}
    val_labels = np.asarray(
        [class_index[sample_by_id[image_id].label] for image_id in val_ids], dtype=np.int64
    )
    return val_ids, val_labels, test_ids, classes


def _input_condition(modality: str, condition: str, base_condition: str) -> str:
    ablations = TEXT_ABLATIONS if modality == "text" else IMAGE_ABLATIONS
    return condition if condition in ablations else base_condition


def _load_pair(
    outputs_dir: Path,
    *,
    image_method: str,
    image_base_condition: str,
    text_method: str,
    text_base_condition: str,
    seed: int,
    split: str,
    condition: str,
    expected_ids: list[str],
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_condition = _input_condition("image", condition, image_base_condition)
    text_condition = _input_condition("text", condition, text_base_condition)
    image_logits = _read_logits(
        preds_path(outputs_dir, image_method, seed, split, image_condition),
        expected_ids,
        n_classes,
    )
    text_logits = _read_logits(
        preds_path(outputs_dir, text_method, seed, split, text_condition),
        expected_ids,
        n_classes,
    )
    return image_logits, text_logits


def _validate_config(cfg: DictConfig) -> tuple[list[int], list[str]]:
    seeds = [int(seed) for seed in cfg.seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")
    conditions = [str(condition) for condition in cfg.conditions]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be a non-empty list without duplicates")
    invalid_conditions = sorted(set(conditions) - set(CONDITIONS))
    if invalid_conditions:
        raise ValueError(
            f"unknown conditions {invalid_conditions}; expected a subset of {list(CONDITIONS)}"
        )
    for key in ("image_method", "image_condition", "text_method", "text_condition"):
        if not str(cfg[key]).strip():
            raise ValueError(f"{key} must be a non-empty string")
    return seeds, conditions


def _meta_path(output_dir: Path, method: str) -> Path:
    return output_dir / method / "meta.json"


def _write_meta(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight_prediction_files(
    outputs_dir: Path,
    *,
    seeds: list[int],
    conditions: list[str],
    image_method: str,
    image_condition: str,
    text_method: str,
    text_condition: str,
    text_ft_method: str | None,
    text_ft_condition: str,
) -> None:
    """Fail before writing any fusion outputs if a required input file is absent."""
    method_pairs = [(text_method, text_condition)]
    if text_ft_method:
        method_pairs.append((text_ft_method, text_ft_condition))
    for seed in seeds:
        for condition in conditions:
            image_source = _input_condition("image", condition, image_condition)
            for split in ("val", "test"):
                image_path = preds_path(outputs_dir, image_method, seed, split, image_source)
                if not image_path.is_file():
                    raise FileNotFoundError(f"missing required image prediction file: {image_path}")
                for method, base_condition in method_pairs:
                    text_source = _input_condition("text", condition, base_condition)
                    text_path = preds_path(outputs_dir, method, seed, split, text_source)
                    if not text_path.is_file():
                        raise FileNotFoundError(
                            f"missing required text prediction file: {text_path}"
                        )


def run(cfg: DictConfig) -> dict[str, Any]:
    """Fuse configured prediction files, calibrating exclusively on validation logits."""
    seeds, conditions = _validate_config(cfg)
    outputs_dir = Path(to_absolute_path(cfg.outputs_dir))
    preds_dir = outputs_dir / "preds"
    upmc_dir = Path(to_absolute_path(cfg.paths.upmc_dir))
    val_ids, val_labels, test_ids, classes = _labels_and_ids(
        upmc_dir, Path(to_absolute_path(cfg.val_ids))
    )
    n_classes = len(classes)
    if val_labels.min() < 0 or val_labels.max() >= n_classes:
        raise ValueError("validation labels are outside the class range")

    temperature_logs = np.linspace(
        TEMPERATURE_LOG_MIN,
        TEMPERATURE_LOG_MAX,
        round((TEMPERATURE_LOG_MAX - TEMPERATURE_LOG_MIN) / TEMPERATURE_LOG_STEP) + 1,
    )
    summaries: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {
        "late_sum": {
            "fusion": "sum of raw modality logits",
            "image_method": str(cfg.image_method),
            "image_condition": str(cfg.image_condition),
            "text_method": str(cfg.text_method),
            "text_condition": str(cfg.text_condition),
            "validation_bias_note": (
                "The validation split was also used for unimodal early stopping and is used "
                "here for calibration and fusion-weight selection; validation scores may be "
                "optimistic. Test data is not used to select temperatures or weights."
            ),
            "conditions": {},
        },
        "late_calibrated": {
            "fusion": "validation-calibrated weighted sum of log probabilities",
            "image_method": str(cfg.image_method),
            "image_condition": str(cfg.image_condition),
            "text_method": str(cfg.text_method),
            "text_condition": str(cfg.text_condition),
            "temperature_log_grid": [TEMPERATURE_LOG_MIN, TEMPERATURE_LOG_MAX],
            "temperature_log_step": TEMPERATURE_LOG_STEP,
            "weight_grid_step": WEIGHT_STEP,
            "conditions": {},
        },
    }

    text_ft_method = cfg.get("text_ft_method")
    text_ft_condition = str(cfg.get("text_ft_condition") or cfg.text_condition)
    _preflight_prediction_files(
        outputs_dir,
        seeds=seeds,
        conditions=conditions,
        image_method=str(cfg.image_method),
        image_condition=str(cfg.image_condition),
        text_method=str(cfg.text_method),
        text_condition=str(cfg.text_condition),
        text_ft_method=str(text_ft_method) if text_ft_method else None,
        text_ft_condition=text_ft_condition,
    )
    if text_ft_method:
        metadata["late_calibrated_ft"] = {
            "fusion": (
                "validation-calibrated weighted sum of image and fine-tuned-text log probabilities"
            ),
            "image_method": str(cfg.image_method),
            "image_condition": str(cfg.image_condition),
            "text_method": str(text_ft_method),
            "text_condition": text_ft_condition,
            "validation_bias_note": (
                "The validation split was also used for unimodal early stopping and is used "
                "here for calibration and fusion-weight selection; validation scores may be "
                "optimistic. Test data is not used to select temperatures or weights."
            ),
            "temperature_log_grid": [TEMPERATURE_LOG_MIN, TEMPERATURE_LOG_MAX],
            "temperature_log_step": TEMPERATURE_LOG_STEP,
            "weight_grid_step": WEIGHT_STEP,
            "conditions": {},
        }

    for seed in seeds:
        for condition in conditions:
            image_val, text_val = _load_pair(
                outputs_dir,
                image_method=str(cfg.image_method),
                image_base_condition=str(cfg.image_condition),
                text_method=str(cfg.text_method),
                text_base_condition=str(cfg.text_condition),
                seed=seed,
                split="val",
                condition=condition,
                expected_ids=val_ids,
                n_classes=n_classes,
            )
            image_test, text_test = _load_pair(
                outputs_dir,
                image_method=str(cfg.image_method),
                image_base_condition=str(cfg.image_condition),
                text_method=str(cfg.text_method),
                text_base_condition=str(cfg.text_condition),
                seed=seed,
                split="test",
                condition=condition,
                expected_ids=test_ids,
                n_classes=n_classes,
            )

            sum_val = (image_val + text_val).astype(np.float32)
            sum_test = (image_test + text_test).astype(np.float32)
            write_preds(
                preds_path(outputs_dir, "late_sum", seed, "val", condition),
                val_ids,
                sum_val,
                n_classes=n_classes,
            )
            write_preds(
                preds_path(outputs_dir, "late_sum", seed, "test", condition),
                test_ids,
                sum_test,
                n_classes=n_classes,
            )

            image_temperature, image_temperature_nll = calibrate_temperature(
                image_val, val_labels, temperature_logs
            )
            text_temperature, text_temperature_nll = calibrate_temperature(
                text_val, val_labels, temperature_logs
            )
            weight, val_top1, val_nll = choose_weight(
                image_val,
                text_val,
                val_labels,
                image_temperature,
                text_temperature,
            )
            calibrated_val = (
                weight * calibrated_log_probs(image_val, image_temperature)
                + (1 - weight) * calibrated_log_probs(text_val, text_temperature)
            ).astype(np.float32)
            calibrated_test = (
                weight * calibrated_log_probs(image_test, image_temperature)
                + (1 - weight) * calibrated_log_probs(text_test, text_temperature)
            ).astype(np.float32)
            write_preds(
                preds_path(outputs_dir, "late_calibrated", seed, "val", condition),
                val_ids,
                calibrated_val,
                n_classes=n_classes,
            )
            write_preds(
                preds_path(outputs_dir, "late_calibrated", seed, "test", condition),
                test_ids,
                calibrated_test,
                n_classes=n_classes,
            )
            selected = {
                "seed": seed,
                "condition": condition,
                "image_temperature": image_temperature,
                "image_temperature_val_nll": image_temperature_nll,
                "text_temperature": text_temperature,
                "text_temperature_val_nll": text_temperature_nll,
                "image_weight": weight,
                "validation_top1": val_top1,
                "validation_nll": val_nll,
            }
            metadata["late_calibrated"]["conditions"].setdefault(str(seed), {})[condition] = (
                selected
            )

            if text_ft_method:
                image_ft_val, text_ft_val = _load_pair(
                    outputs_dir,
                    image_method=str(cfg.image_method),
                    image_base_condition=str(cfg.image_condition),
                    text_method=str(text_ft_method),
                    text_base_condition=text_ft_condition,
                    seed=seed,
                    split="val",
                    condition=condition,
                    expected_ids=val_ids,
                    n_classes=n_classes,
                )
                image_ft_test, text_ft_test = _load_pair(
                    outputs_dir,
                    image_method=str(cfg.image_method),
                    image_base_condition=str(cfg.image_condition),
                    text_method=str(text_ft_method),
                    text_base_condition=text_ft_condition,
                    seed=seed,
                    split="test",
                    condition=condition,
                    expected_ids=test_ids,
                    n_classes=n_classes,
                )
                image_ft_temperature, image_ft_temperature_nll = calibrate_temperature(
                    image_ft_val, val_labels, temperature_logs
                )
                text_ft_temperature, text_ft_temperature_nll = calibrate_temperature(
                    text_ft_val, val_labels, temperature_logs
                )
                ft_weight, ft_top1, ft_nll = choose_weight(
                    image_ft_val,
                    text_ft_val,
                    val_labels,
                    image_ft_temperature,
                    text_ft_temperature,
                )
                ft_val = (
                    ft_weight * calibrated_log_probs(image_ft_val, image_ft_temperature)
                    + (1 - ft_weight) * calibrated_log_probs(text_ft_val, text_ft_temperature)
                ).astype(np.float32)
                ft_test = (
                    ft_weight * calibrated_log_probs(image_ft_test, image_ft_temperature)
                    + (1 - ft_weight) * calibrated_log_probs(text_ft_test, text_ft_temperature)
                ).astype(np.float32)
                write_preds(
                    preds_path(outputs_dir, "late_calibrated_ft", seed, "val", condition),
                    val_ids,
                    ft_val,
                    n_classes=n_classes,
                )
                write_preds(
                    preds_path(outputs_dir, "late_calibrated_ft", seed, "test", condition),
                    test_ids,
                    ft_test,
                    n_classes=n_classes,
                )
                metadata["late_calibrated_ft"]["conditions"].setdefault(str(seed), {})[
                    condition
                ] = {
                    "seed": seed,
                    "condition": condition,
                    "image_temperature": image_ft_temperature,
                    "image_temperature_val_nll": image_ft_temperature_nll,
                    "text_temperature": text_ft_temperature,
                    "text_temperature_val_nll": text_ft_temperature_nll,
                    "image_weight": ft_weight,
                    "validation_top1": ft_top1,
                    "validation_nll": ft_nll,
                }

            log.info(
                "seed %d %s: calibrated w=%.2f, val top-1 %.4f, NLL %.4f",
                seed,
                condition,
                weight,
                val_top1,
                val_nll,
            )
        summaries[str(seed)] = metadata["late_calibrated"]["conditions"][str(seed)]

    for method, method_metadata in metadata.items():
        _write_meta(_meta_path(preds_dir, method), method_metadata)
    return summaries


@hydra.main(version_base=None, config_path="../../../configs", config_name="late_fusion")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
