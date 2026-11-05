import logging
from pathlib import Path

from PIL import Image

from recipe_fusion.data.upmc import SPLITS, read_split

log = logging.getLogger(__name__)


def check_images(root: Path) -> list[tuple[Path, str]]:
    """Return image paths and errors for files that cannot be verified."""
    unreadable = []

    for split in SPLITS:
        for sample in read_split(root, split):
            path = sample.image_path(root)
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
                unreadable.append((path, str(exc)))

    return unreadable
