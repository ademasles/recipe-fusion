"""Helpers shared by the text baselines and the evaluation script."""

import html


def mentions_class(title: str, label: str) -> bool:
    """True if the class name appears in the title.

    The class name has its underscores replaced by spaces and is lowercased; the title is
    html-unescaped and lowercased first (same preprocessing as the text features).
    """
    name = label.replace("_", " ").lower()
    return name in html.unescape(title).lower()
