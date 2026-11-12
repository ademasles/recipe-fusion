"""Title cleaning shared by every text model (TF-IDF, frozen and fine-tuned BERT).

About 4 % of the raw titles carry encoding debris: entities escaped twice
(``Mom&amp;rsquo;s``) and UTF-8 bytes written out as text (``\\xe2\\x84\\xa2`` for ``™``).
Every text model must see the same cleaned string, so they all call :func:`clean_title`.
"""

import html
import re
import unicodedata

_ESCAPED_BYTES = re.compile(r"(?:\\x[0-9a-fA-F]{2})+")
_MAX_UNESCAPE_PASSES = 3


def _decode_bytes(match: re.Match[str]) -> str:
    raw = bytes.fromhex(match.group(0).replace("\\x", ""))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return match.group(0)  # not valid UTF-8: leave it as written


def clean_title(title: str) -> str:
    """Unescape HTML entities until stable, decode literal UTF-8 bytes, collapse spaces."""
    for _ in range(_MAX_UNESCAPE_PASSES):
        unescaped = html.unescape(title)
        if unescaped == title:
            break
        title = unescaped
    title = _ESCAPED_BYTES.sub(_decode_bytes, title)
    return " ".join(title.split())


def mentions_class(title: str, label: str) -> bool:
    """Whether the class name (``apple_pie`` -> ``apple pie``) appears in the cleaned title.

    Case and accents are ignored, so ``Crème Brûlée`` mentions ``creme_brulee``. Titles
    without it form the ``no_class_name`` subset reported by ``evaluate``.
    """
    return label.replace("_", " ").lower() in _fold(clean_title(title))


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))
