"""Image integrity checks for the UPMC Food-101 dataset."""

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from recipe_fusion.data.upmc import SPLITS, read_split

log = logging.getLogger(__name__)

ImageStatus = Literal["missing", "unreadable"]


@dataclass(frozen=True)
class ImageIssue:
    path: Path
    status: ImageStatus
    error: str


def check_images(root: Path, splits: Iterable[str] = SPLITS) -> list[ImageIssue]:
    """Return images that are missing or cannot be fully decoded."""
    issues = []
    checked = 0

    for split in splits:
        for sample in tqdm(read_split(root, split), desc=f"Checking {split} images", unit="image"):
            checked += 1
            path = sample.image_path(root)
            try:
                with Image.open(path) as image:
                    image.load()
            except FileNotFoundError as exc:
                issues.append(ImageIssue(path, "missing", str(exc)))
            except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
                issues.append(ImageIssue(path, "unreadable", str(exc)))

    unreadable = sum(issue.status == "unreadable" for issue in issues)
    missing = sum(issue.status == "missing" for issue in issues)
    log.info(
        "checked %d images: %d unreadable, %d missing",
        checked,
        unreadable,
        missing,
    )
    for issue in issues[:5]:
        log.warning("%s image: %s (%s)", issue.status, issue.path, issue.error)

    return issues


def write_report(issues: list[ImageIssue], output: Path) -> None:
    """Write all image issues as a CSV report."""
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "status", "error"))
        writer.writerows((issue.path, issue.status, issue.error) for issue in issues)


@hydra.main(version_base=None, config_path="../../../configs", config_name="check_images")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.upmc_dir)
    issues = check_images(root)
    report = Path(HydraConfig.get().runtime.output_dir) / "image_issues.csv"
    write_report(issues, report)
    log.info("image integrity report: %s (%d issues)", report, len(issues))
    if issues:
        raise RuntimeError(f"found {len(issues)} image integrity issue(s); see {report}")


if __name__ == "__main__":
    main()
