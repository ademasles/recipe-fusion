import pytest

from recipe_fusion.data.clean import clean_title


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  Bob&#039;s Red Mill ", "Bob's Red Mill"),
        ("Raw Apple Pie &amp; Crumble", "Raw Apple Pie & Crumble"),
        ("Mom&amp;rsquo;s Apple Pie", "Mom\N{RIGHT SINGLE QUOTATION MARK}s Apple Pie"),
        ("Know \\xe2\\x84\\xa2 Cake", "Know ™ Cake"),
        ("Cr\\xc3\\xa8me br\\xc3\\xbbl\\xc3\\xa9e", "Crème brûlée"),
        ("Broken \\xff byte", "Broken \\xff byte"),
        ("Too   many\tspaces", "Too many spaces"),
        ("Plain title", "Plain title"),
    ],
)
def test_clean_title(raw: str, clean: str) -> None:
    assert clean_title(raw) == clean


def test_clean_title_is_idempotent() -> None:
    raw = "Mom&amp;amp;rsquo;s \\xe2\\x84\\xa2"
    assert clean_title(clean_title(raw)) == clean_title(raw)
