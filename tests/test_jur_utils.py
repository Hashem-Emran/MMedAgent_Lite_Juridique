"""Unit tests for shared utilities (language detection, mojibake, normalization)."""
from __future__ import annotations

from dataset.jur_utils import (
    arabic_ratio,
    clean_text,
    dedup_key,
    detect_language,
    fix_mojibake,
    is_placeholder_answer,
    normalize_arabic,
    normalize_fr,
)


class TestArabicRatio:
    def test_pure_arabic(self):
        assert arabic_ratio("ما هو الدستور") >= 0.9

    def test_pure_french(self):
        assert arabic_ratio("Quel est le rôle du gouvernement") == 0.0

    def test_empty(self):
        assert arabic_ratio("") == 0.0

    def test_digits_only(self):
        assert arabic_ratio("1958 2011") == 0.0


class TestLanguageDetection:
    def test_arabic_question(self):
        assert detect_language("ما هي مدة الولاية؟", threshold=0.30) == "ar"

    def test_french_question(self):
        assert detect_language("Quelle est la durée du mandat ?", threshold=0.30) == "fr"

    def test_mixed_below_threshold_returns_fr(self):
        mixed = "Selon la loi n° 2011 ما"
        assert detect_language(mixed, threshold=0.30) == "fr"


class TestMojibake:
    def test_cp1252_double_encoded_french(self):
        broken = "RÃ©glementation prÃ©vue"
        fixed = fix_mojibake(broken)
        assert "é" in fixed and "Ã" not in fixed

    def test_clean_text_unchanged(self):
        assert fix_mojibake("Réglementation") == "Réglementation"

    def test_empty_passthrough(self):
        assert fix_mojibake("") == ""


class TestNormalizeArabic:
    def test_alef_variants_collapsed(self):
        assert normalize_arabic("أحمد إلى آدم", remove_tashkeel=True, normalize_ta_marbuta=False) == "احمد الي ادم"

    def test_ya_collapsed(self):
        assert "ى" not in normalize_arabic("على", remove_tashkeel=True, normalize_ta_marbuta=False)

    def test_tashkeel_removed(self):
        assert normalize_arabic("مَكْتَبَة", remove_tashkeel=True, normalize_ta_marbuta=False) == "مكتبة"

    def test_tashkeel_preserved_when_disabled(self):
        result = normalize_arabic("مَكْتَبَة", remove_tashkeel=False, normalize_ta_marbuta=False)
        assert "َ" in result


class TestCleanText:
    def test_strips_html_tags(self):
        assert clean_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_strips_html_entities(self):
        assert clean_text("A&amp;B&nbsp;C") == "A B C"

    def test_collapses_whitespace(self):
        assert clean_text("a   b\t\tc\n\nd") == "a b c d"


class TestPlaceholderAnswer:
    def test_detects_na(self):
        assert is_placeholder_answer("N/A")

    def test_detects_empty(self):
        assert is_placeholder_answer("")

    def test_detects_dash(self):
        assert is_placeholder_answer("-")

    def test_real_answer_passes(self):
        assert not is_placeholder_answer("Le Parlement est bicaméral.")


class TestDedupKey:
    def test_same_question_same_doc_same_key(self):
        k1 = dedup_key("Quelle est la durée ?", "12345")
        k2 = dedup_key("Quelle est la durée ?", "12345")
        assert k1 == k2

    def test_different_doc_different_key(self):
        k1 = dedup_key("Quelle est la durée ?", "12345")
        k2 = dedup_key("Quelle est la durée ?", "67890")
        assert k1 != k2

    def test_whitespace_insensitive(self):
        k1 = dedup_key("Quelle   est   la    durée ?", "12345")
        k2 = dedup_key("Quelle est la durée ?", "12345")
        assert k1 == k2

    def test_case_insensitive(self):
        assert dedup_key("DURÉE", "1") == dedup_key("durée", "1")


class TestNormalizeFr:
    def test_nfc_composition(self):
        decomposed = "cafe\u0301"
        assert normalize_fr(decomposed) == "café"
