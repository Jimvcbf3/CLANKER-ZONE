import re
from typing import Optional, Tuple


# UI pre-clean regexes/sets
_RE_PUNCT_ONLY = re.compile(r"^\s*[\[\]\(\):;.,\-\+\|\!~/\\_=:<>]+(\s*[\[\]\(\):;.,\-\+\|\!~/\\_=:<>]+)*\s*$")
_RE_FAKE_BULLET = re.compile(r"^\s*\[\s*\]\s*")
_RE_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_RE_PERCENT = re.compile(r"\b\d{1,3}%\b")
_PUNCT_SET = set("[](){}:;.,-+|!~/\\_=<>")


def _punct_ratio(s: str) -> float:
    if not s:
        return 1.0
    total = sum(1 for c in s if not c.isspace()) or 1
    p = sum(1 for c in s if c in _PUNCT_SET)
    return p / float(total)


def _has_ui_text(s: str) -> bool:
    for ch in s:
        if ch.isdigit() or ("a" <= ch.lower() <= "z") or ("\u4e00" <= ch <= "\u9fff"):
            return True
    return False


def preclean_ui(text: str) -> Tuple[str, int]:
    dropped = 0
    out_lines = []
    for raw in (text or "").splitlines():
        ln = raw.rstrip("\r\n")
        if not ln.strip():
            continue
        if _RE_TIME.search(ln) or _RE_PERCENT.search(ln):
            out_lines.append(ln.strip())
            continue
        ln2 = _RE_FAKE_BULLET.sub("", ln).strip()
        if _RE_PUNCT_ONLY.match(ln2 or ln):
            dropped += 1
            continue
        if _punct_ratio(ln2) > 0.6:
            dropped += 1
            continue
        if not _has_ui_text(ln2):
            dropped += 1
            continue
        out_lines.append(ln2)
    return ("\n".join(out_lines), dropped)


class OcrCleaner:
    def __init__(
        self,
        preserve_decimals: bool = True,
        collapse_whitespace: bool = True,
        normalize_dashes: bool = True,
        strip_ansi: bool = True,
    ) -> None:
        self.preserve_decimals = preserve_decimals
        self.collapse_whitespace = collapse_whitespace
        self.normalize_dashes = normalize_dashes
        self.strip_ansi = strip_ansi

        # Precompile regexes
        self._re_ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
        self._re_space = re.compile(r"[ \t\u00A0\u2000-\u200B]+")
        # Surround punctuation with spaces except for decimal points within numbers
        self._re_punct = re.compile(r"([\,;:!?])(?!\d)")
        # Normalize various dash/hyphen characters
        self._re_dash = re.compile(r"[\u2012\u2013\u2014\u2015\u2212]")

    def _strip_ansi(self, s: str) -> str:
        return self._re_ansi.sub("", s)

    def _normalize_dashes(self, s: str) -> str:
        return self._re_dash.sub("-", s)

    def _safe_punct_spacing(self, s: str) -> str:
        s = self._re_punct.sub(r"\1 ", s)
        return s

    def _collapse_whitespace(self, s: str) -> str:
        lines = s.splitlines()
        cleaned_lines = []
        for ln in lines:
            ln = self._re_space.sub(" ", ln).strip()
            cleaned_lines.append(ln)
        return "\n".join(cleaned_lines).strip()

    def clean(self, text: str) -> str:
        s = text or ""
        if self.strip_ansi:
            s = self._strip_ansi(s)
        if self.normalize_dashes:
            s = self._normalize_dashes(s)
        if self.preserve_decimals:
            s = self._safe_punct_spacing(s)
        if self.collapse_whitespace:
            s = self._collapse_whitespace(s)
        return s


def clean_text(text: Optional[str]) -> str:
    return OcrCleaner().clean(text or "")


def clean(text: Optional[str]) -> str:
    return OcrCleaner().clean(text or "")

