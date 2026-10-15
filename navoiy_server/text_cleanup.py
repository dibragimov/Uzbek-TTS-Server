"""Remove LLM formatting that should never be spoken.

Dograh's Speaches path only strips function-call XML, so markdown and emoji
from the LLM reach the TTS. This runs before RC4's telecom normalizer and is
deliberately narrow: USSD codes (*102#), Uzbek apostrophes (oʻ, gʻ, o', g'),
numbers and punctuation pass through untouched.
"""
from __future__ import annotations

import re

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF"   # pictographs, emoticons, transport, symbols & pictographs ext.
    "\u2600-\u27BF"            # misc symbols, dingbats
    "\u2B00-\u2BFF"            # arrows/stars used as emoji
    "\uFE0E\uFE0F\u200D"       # variation selectors, zero-width joiner
    "\U000E0020-\U000E007F]+"  # tag characters (flag sequences)
)
_HEADER = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+")
_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*•▪◦–—]|\d{1,2}[.)])[ \t]+")
# Only PAIRED markers around plain text. Never touch backticks (Uzbek is often
# typed "o`zbek") or unpaired asterisks (USSD/GSM codes like *102# or **21*...#).
_EMPHASIS = re.compile(r"(\*\*|__|~~)([^*#_~\n]+?)\1")
_LINK = re.compile(r"\[([^\]\n]+)\]\((?:[^)\s]+)\)")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_NEWLINES = re.compile(r"\s*\n+\s*")


def clean(text: str) -> str:
    out = _LINK.sub(r"\1", text)          # [label](url) -> label
    out = _EMPHASIS.sub(r"\2", out)      # **bold**, __bold__, ~~strike~~
    out = _HEADER.sub("", out)            # "## Title" at line start
    out = _BULLET.sub("", out)            # "- item", "* item", "1. item" at line start
    out = _EMOJI.sub(" ", out)
    out = _NEWLINES.sub(" ", out)
    out = _SPACES.sub(" ", out).strip()
    return out


_UZ_CYRILLIC = re.compile(r"[ЎўҚқҒғҲҳ]")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def looks_russian(text: str) -> bool:
    """Cyrillic text without Uzbek-only letters: Navoiy cannot speak Russian."""
    return bool(_CYRILLIC.search(text)) and not _UZ_CYRILLIC.search(text)
