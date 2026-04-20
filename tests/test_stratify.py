"""Unit tests for stratification features and composite scoring."""
from __future__ import annotations

import pandas as pd
import pytest

from dataset.stratify_jur import (
    assign_difficulty,
    composite_score,
    f1_ans_len,
    f2_ctx_len,
    f3_lexical_overlap,
    f4_has_negation,
    f5_has_quantifier,
    f6_cross_refs,
    f7_question_type,
    stratified_split,
)


WEIGHTS = {
    "f1_ans_len": 0.20, "f2_ctx_len": 0.15, "f3_lexical_overlap": 0.25,
    "f4_has_negation": 0.10, "f5_has_quantifier": 0.05,
    "f6_cross_refs": 0.10, "f7_question_type": 0.15,
}


class TestLengthFeatures:
    def test_f1_saturates_at_1(self):
        assert f1_ans_len(500) == 1.0
        assert f1_ans_len(10000) == 1.0

    def test_f1_zero_for_empty(self):
        assert f1_ans_len(0) == 0.0

    def test_f2_saturates(self):
        assert f2_ctx_len(8000) == 1.0
        assert f2_ctx_len(4000) == 0.5


class TestLexicalOverlap:
    def test_identical_strings_zero(self):
        assert f3_lexical_overlap("hello world", "hello world") == 0.0

    def test_disjoint_strings_one(self):
        assert f3_lexical_overlap("abc def", "xyz uvw") == 1.0

    def test_partial_overlap(self):
        val = f3_lexical_overlap("a b c d", "a b e f")
        assert 0 < val < 1


class TestNegation:
    def test_french_ne_pas(self):
        assert f4_has_negation("Le Roi ne peut pas dissoudre", "fr") == 0.5

    def test_french_without(self):
        assert f4_has_negation("Sans préavis, que se passe-t-il ?", "fr") == 0.5

    def test_french_no_negation(self):
        assert f4_has_negation("Quelle est la durée ?", "fr") == 0.0

    def test_arabic_negation(self):
        assert f4_has_negation("لا يجوز الطعن", "ar") == 0.5

    def test_arabic_no_negation(self):
        assert f4_has_negation("ما هي مدة الولاية", "ar") == 0.0


class TestQuantifier:
    def test_french_number(self):
        assert f5_has_quantifier("Combien de ministres sont nommés ?", "fr") == 0.3

    def test_french_tous(self):
        assert f5_has_quantifier("Tous les citoyens sont égaux", "fr") == 0.3

    def test_french_none(self):
        assert f5_has_quantifier("Quelle est la procédure ?", "fr") == 0.0

    def test_arabic_all(self):
        assert f5_has_quantifier("كل المواطنين متساوون", "ar") == 0.3


class TestCrossRefs:
    def test_french_article(self):
        val = f6_cross_refs("selon article 15", "texte", "fr")
        assert val > 0

    def test_arabic_article(self):
        val = f6_cross_refs("حسب المادة 5", "نص", "ar")
        assert val > 0

    def test_saturates(self):
        text = " ".join(f"article {i}" for i in range(20))
        assert f6_cross_refs(text, "", "fr") == 1.0


class TestQuestionType:
    def test_reasoning_french(self):
        assert f7_question_type("Pourquoi le régime est-il parlementaire ?", "fr") == 0.9

    def test_descriptive_french(self):
        assert f7_question_type("Quelle est la capitale ?", "fr") == 0.5

    def test_factual_french(self):
        assert f7_question_type("Quand a été promulguée la constitution ?", "fr") == 0.1

    def test_reasoning_arabic(self):
        assert f7_question_type("لماذا يوجد مجلسان", "ar") == 0.9

    def test_factual_arabic(self):
        assert f7_question_type("متى تم نشر الظهير", "ar") == 0.1


class TestCompositeScore:
    def test_score_in_unit_interval(self):
        features = pd.DataFrame({
            "f1_ans_len": [0.2, 1.0],
            "f2_ctx_len": [0.1, 0.8],
            "f3_lexical_overlap": [0.5, 0.9],
            "f4_has_negation": [0.0, 0.5],
            "f5_has_quantifier": [0.0, 0.3],
            "f6_cross_refs": [0.0, 0.7],
            "f7_question_type": [0.1, 0.9],
        })
        scores = composite_score(features, WEIGHTS)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_zero_features_zero_score(self):
        features = pd.DataFrame({k: [0.0] for k in WEIGHTS})
        assert composite_score(features, WEIGHTS).iloc[0] == 0.0


class TestAssignDifficulty:
    def test_tertile_partition(self):
        scores = pd.Series(list(range(99)), dtype=float) / 99
        labels, th = assign_difficulty(scores)
        counts = labels.value_counts()
        assert abs(counts["easy"] - 33) <= 2
        assert abs(counts["medium"] - 33) <= 2
        assert abs(counts["hard"] - 33) <= 2
        assert th["easy"] < th["medium"]


class TestStratifiedSplit:
    def test_ratio_approximately_80_20(self):
        df = pd.DataFrame({
            "doc_type": ["Dahir"] * 100,
            "doc_date": ["2000-01-01"] * 100,
        })
        assignments = stratified_split(df, train_frac=0.8, seed=42)
        test_count = (assignments == "test").sum()
        assert 18 <= test_count <= 22

    def test_deterministic(self):
        df = pd.DataFrame({
            "doc_type": ["Dahir"] * 50 + ["Décret"] * 50,
            "doc_date": ["1960-01-01"] * 50 + ["2000-01-01"] * 50,
        })
        a = stratified_split(df, train_frac=0.8, seed=42)
        b = stratified_split(df, train_frac=0.8, seed=42)
        pd.testing.assert_series_equal(a, b)
