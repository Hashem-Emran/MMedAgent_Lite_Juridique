"""Shared utilities for the Jurida D1+D2 pipeline (FR+AR preprocessing, stratification)."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

ARABIC_RANGE = (0x0600, 0x06FF)
TASHKEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670\u0640]")
ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"})
YA_NORMALIZE = str.maketrans({"ى": "ي"})
TA_MARBUTA = str.maketrans({"ة": "ه"})
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITY_PATTERN = re.compile(r"&(?:amp|nbsp|lt|gt|quot|apos|#\d+);")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
PLACEHOLDER_ANSWERS = {"", "n/a", "na", "-", "--", "none", "null", "nan"}


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    lo, hi = ARABIC_RANGE
    arabic = sum(1 for c in text if lo <= ord(c) <= hi)
    letters = sum(1 for c in text if c.isalpha())
    return arabic / letters if letters else 0.0


def detect_language(text: str, threshold: float) -> str:
    return "ar" if arabic_ratio(text) >= threshold else "fr"


def fix_mojibake(text: str) -> str:
    """Recover UTF-8 text that was double-encoded as cp1252."""
    if not text:
        return text
    if "�" in text or "Ã" in text or "Â" in text:
        try:
            return text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                import ftfy
                return ftfy.fix_text(text)
            except ImportError:
                return text
    return text


def normalize_arabic(text: str, remove_tashkeel: bool, normalize_ta_marbuta: bool) -> str:
    if not text:
        return text
    if remove_tashkeel:
        text = TASHKEEL_PATTERN.sub("", text)
    text = text.translate(ALEF_VARIANTS).translate(YA_NORMALIZE)
    if normalize_ta_marbuta:
        text = text.translate(TA_MARBUTA)
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)
    text = CONTROL_CHARS.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_fr(text: str) -> str:
    return unicodedata.normalize("NFC", text) if text else text


def is_placeholder_answer(text: str) -> bool:
    if text is None:
        return True
    stripped = text.strip().lower()
    return stripped in PLACEHOLDER_ANSWERS


def dedup_key(question: str, doc_id: str) -> str:
    q = WHITESPACE_PATTERN.sub(" ", (question or "").lower().strip())
    return hashlib.sha1(f"{doc_id}||{q}".encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def get_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def md5_of_file(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
