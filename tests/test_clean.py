import pytest

from recipe_fusion.data.clean import mentions_class


@pytest.mark.parametrize(
    ("title", "label", "expected"),
    [
        ("Apple Pie Recipe", "apple_pie", True),
        ("Grandma's dessert", "apple_pie", False),
        ("Grandma&#039;s APPLE PIE", "apple_pie", True),  # unescape + lowercase
        ("Easy Apple-Pie", "apple_pie", False),  # exact name only: a hyphen is not a space
        ("Caesar Salad with croutons", "caesar_salad", True),
        ("", "apple_pie", False),
    ],
)
def test_mentions_class(title: str, label: str, expected: bool) -> None:
    assert mentions_class(title, label) is expected
