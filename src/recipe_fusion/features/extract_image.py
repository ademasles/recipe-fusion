"""Extract pooled and spatial ViT features from UPMC Food-101 images."""

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import hydra
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from recipe_fusion.data.upmc import SPLITS, Sample, read_split
from recipe_fusion.features.cache import write_features
from recipe_fusion.features.text import git_commit
from recipe_fusion.seed import pick_device

log = logging.getLogger(__name__)

ENCODER_NAME = "vit_b_16_swag"
FEATURE_DIM = 768
GRID_SIZE = 8
PATCH_GRID_SIZE = 24
TOKENS_PER_IMAGE = GRID_SIZE * GRID_SIZE


class ImageDataset(Dataset[Tensor]):
    def __init__(
        self,
        root: Path,
        samples: list[Sample],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tensor:
        sample = self.samples[index]
        with Image.open(sample.image_path(self.root)) as image:
            rgb_image = image.convert("RGB")
            try:
                return self.transform(rgb_image)
            finally:
                rgb_image.close()


class GridWriter:
    def __init__(self, path: Path, sample_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        self.path = path
        self.array: np.memmap = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(sample_count, TOKENS_PER_IMAGE, FEATURE_DIM),
        )
        self.offset = 0

    def __call__(self, sequence: Tensor) -> None:
        if sequence.ndim != 3 or sequence.shape[1:] != (PATCH_GRID_SIZE**2 + 1, FEATURE_DIM):
            raise ValueError(f"expected encoder output (N, 577, 768), got {tuple(sequence.shape)}")

        patches = sequence[:, 1:, :].permute(0, 2, 1)
        patches = patches.reshape(-1, FEATURE_DIM, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
        grid = functional.avg_pool2d(patches, kernel_size=3, stride=3)
        grid = grid.permute(0, 2, 3, 1).reshape(-1, TOKENS_PER_IMAGE, FEATURE_DIM)
        values = grid.to(dtype=torch.float16, device="cpu").numpy()
        end = self.offset + len(values)
        if end > len(self.array):
            raise ValueError("encoder produced more images than the dataset contains")
        self.array[self.offset : end] = values
        self.offset = end

    def finish(self) -> npt.NDArray[np.float16]:
        if self.offset != len(self.array):
            raise ValueError(f"encoder produced {self.offset} of {len(self.array)} expected images")
        self.array.flush()
        return self.array

    def close(self, *, remove: bool = False) -> None:
        self.array.flush()
        del self.array
        if remove:
            self.path.unlink(missing_ok=True)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def encode(
    loader: Iterable[Tensor],
    model: nn.Module,
    *,
    device: torch.device | None = None,
    token_callback: Callable[[Tensor], None] | None = None,
) -> npt.NDArray[np.float16]:
    """Encode batches to pooled features; optionally stream encoder token sequences."""
    device = device or _model_device(model)
    model.to(device)
    model.eval()
    hook = None
    if token_callback is not None:
        encoder = getattr(model, "encoder", None)
        if not isinstance(encoder, nn.Module):
            raise TypeError("model must expose an encoder module to extract spatial tokens")

        def capture_tokens(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, Tensor):
                raise TypeError("encoder output must be a tensor")
            token_callback(output)

        hook = encoder.register_forward_hook(capture_tokens)

    pooled_batches: list[npt.NDArray[np.float16]] = []
    try:
        with torch.inference_mode():
            for images in tqdm(loader, desc="Encoding images", unit="batch"):
                images = images.to(device, non_blocking=device.type == "cuda")
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    features = model(images)
                if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                    raise ValueError(
                        f"expected pooled output (N, 768), got {tuple(features.shape)}"
                    )
                pooled_batches.append(features.to(device="cpu", dtype=torch.float16).numpy())
    finally:
        if hook is not None:
            hook.remove()

    if not pooled_batches:
        raise ValueError("cannot extract features from an empty dataset")
    return np.concatenate(pooled_batches, axis=0)


def extract_split(
    root: Path,
    samples: list[Sample],
    model: nn.Module,
    transform: Callable[[Image.Image], Tensor],
    output: Path,
    split: str,
    *,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> int:
    """Extract one split, retrying with smaller batches after CUDA OOM."""
    current_batch_size = batch_size
    ids = [sample.image_name for sample in samples]
    token_path = output / f"{split}_tokens.npy"
    ids_path = output / f"{split}_ids.txt"
    ids_path.unlink(missing_ok=True)

    while True:
        grid_writer: GridWriter | None = None
        try:
            grid_writer = GridWriter(token_path, len(samples))
            dataset = ImageDataset(root, samples, transform)
            loader = DataLoader(
                dataset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            started = time.perf_counter()
            pooled = encode(loader, model, device=device, token_callback=grid_writer)
            tokens = grid_writer.finish()
            mask = np.ones((len(samples), TOKENS_PER_IMAGE), dtype=np.bool_)
            elapsed = time.perf_counter() - started
            metadata = {
                "format": "2.2",
                "encoder": ENCODER_NAME,
                "weights": "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1",
                "d": FEATURE_DIM,
                "L": TOKENS_PER_IMAGE,
                "pooled": "CLS token after the final LayerNorm",
                "grid": f"{GRID_SIZE}x{GRID_SIZE} avg-pool of the "
                f"{PATCH_GRID_SIZE}x{PATCH_GRID_SIZE} patch tokens",
                "preprocessing": "weights.transforms()",
                "autocast": "bfloat16" if device.type == "cuda" else "none",
                "dtype": "float16",
                "git_commit": git_commit(),
                "device": str(device),
                "extraction_seconds": round(elapsed, 2),
                "batch_size": current_batch_size,
                "num_workers": num_workers,
                "n": len(samples),
            }
            write_features(output, split, ids, pooled, tokens=tokens, mask=mask, meta=metadata)
            grid_writer.close()
            log.info(
                "%s: extracted %d images with batch size %d in %.1f seconds",
                split,
                len(samples),
                current_batch_size,
                elapsed,
            )
            return current_batch_size
        except torch.cuda.OutOfMemoryError:
            if grid_writer is not None:
                grid_writer.close(remove=True)
            token_path.unlink(missing_ok=True)
            if device.type != "cuda" or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            log.warning(
                "CUDA out of memory; retrying %s with batch size %d", split, current_batch_size
            )
        except BaseException:
            if grid_writer is not None:
                grid_writer.close(remove=True)
            raise


def load_encoder() -> tuple[nn.Module, Callable[[Image.Image], Tensor]]:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_b_16(weights=weights)
    model.add_module("heads", nn.Identity())
    return model, weights.transforms()


def extract(cfg: DictConfig) -> None:
    """Extract every split. Run ``just check-images`` first: a bad image stops the run."""
    root = Path(to_absolute_path(cfg.paths.upmc_dir))
    output = Path(to_absolute_path(cfg.paths.features_dir)) / "image" / ENCODER_NAME
    device = pick_device(str(cfg.extract.device))
    model, transform = load_encoder()
    for split in SPLITS:
        samples = read_split(root, split)
        extract_split(
            root,
            samples,
            model,
            transform,
            output,
            split,
            device=device,
            batch_size=int(cfg.extract.batch_size),
            num_workers=int(cfg.extract.num_workers),
            pin_memory=bool(cfg.extract.pin_memory),
        )


@hydra.main(version_base=None, config_path="../../../configs", config_name="features/image")
def main(cfg: DictConfig) -> None:
    extract(cfg)


if __name__ == "__main__":
    main()
