import csv
from pathlib import Path

from PIL import Image

from recipe_fusion.features.image import check_images


def test_check_images_finds_truncated_and_missing_images(tmp_path: Path) -> None:
    valid_path = tmp_path / "images/train/dish/valid.jpg"
    truncated_path = tmp_path / "images/train/dish/truncated.jpg"
    valid_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (16, 16), color="red")
    image.save(valid_path, format="JPEG")
    image_data = valid_path.read_bytes()
    truncated_path.write_bytes(image_data[: len(image_data) // 2])

    titles = tmp_path / "texts/train_titles.csv"
    titles.parent.mkdir(parents=True)
    titles.write_text(
        "valid.jpg,Valid,dish\ntruncated.jpg,Truncated,dish\nmissing.jpg,Missing,dish\n",
        encoding="utf-8",
    )
    issues = check_images(tmp_path, splits=("train",))

    assert len(issues) == 2
    assert {issue.path.name: issue.status for issue in issues} == {
        "truncated.jpg": "unreadable",
        "missing.jpg": "missing",
    }


def test_write_report_creates_csv(tmp_path: Path) -> None:
    from recipe_fusion.features.image import ImageIssue, write_report

    report = tmp_path / "image_issues.csv"
    write_report([ImageIssue(tmp_path / "missing.jpg", "missing", "not found")], report)

    with report.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["image", "status", "error"],
        [str(tmp_path / "missing.jpg"), "missing", "not found"],
    ]
