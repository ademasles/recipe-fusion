"""Download UPMC Food-101 from Kaggle, then check that every title has its image.

Run with ``just download`` (about 8.3 GB). ``just download data.texts_only=true`` fetches only
the two title files (7.6 MB), enough to explore the text modality.
"""

import logging
from pathlib import Path

import hydra
import kagglehub
from omegaconf import DictConfig

from recipe_fusion.data.upmc import SPLITS, missing_images, read_split, summary

log = logging.getLogger(__name__)

TEXT_FILES = [f"texts/{split}_titles.csv" for split in SPLITS]


def download(cfg: DictConfig) -> Path:
    root = Path(cfg.paths.upmc_dir)
    root.mkdir(parents=True, exist_ok=True)
    handle = f"{cfg.data.kaggle_handle}/versions/{cfg.data.version}"
    if cfg.data.texts_only:
        for file in TEXT_FILES:
            kagglehub.dataset_download(handle, path=file, output_dir=str(root))
    else:
        kagglehub.dataset_download(handle, output_dir=str(root))
    for split in SPLITS:
        samples = read_split(root, split)
        log.info("%s: %s", split, summary(samples))
        if not cfg.data.texts_only:
            missing = missing_images(root, samples)
            if missing:
                log.warning(
                    "%s: %d titles without an image, e.g. %s",
                    split,
                    len(missing),
                    missing[0].image_path(root),
                )
    log.info("data in %s", root)
    return root


@hydra.main(version_base=None, config_path="../../../configs", config_name="download")
def main(cfg: DictConfig) -> None:
    download(cfg)


if __name__ == "__main__":
    main()
