"""D2 — Stratify FR+AR Q&A into Easy/Medium/Hard buckets + 80/20 train/test.

Computes 7 heuristic difficulty features (answer length, context length, lexical
overlap answer↔context, negation, quantifier, cross-references, question type),
aggregates them into a composite score with config weights, partitions each
language into tertiles, then within each (lang, bucket) splits 80/20 train/test
stratified by doc_type and doc_date decade.

Inputs  : data/interim/qa_fr.parquet, data/interim/qa_ar.parquet
Outputs :
  - data/interim/qa_fr.parquet  (updated: adds difficulty, difficulty_score, split)
  - data/interim/qa_ar.parquet  (updated: same)
  - data/interim/splits.json    (bucket assignments + thresholds + config hash)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.jur_utils import ROOT, ensure_dir, load_config, setup_logger

log = setup_logger("stratify_jur")

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

NEGATION_TERMS_FR = (
    r"\bne\s+\w+\s+pas\b",
    r"\bn[e']\s*\w+\s+pas\b",
    r"\bsans\b",
    r"\baucun(?:e|s)?\b",
    r"\bjamais\b",
    r"\bni\b",
)
NEGATION_TERMS_AR = ("لا ", "لم ", "لن ", "ليس", "غير ", "بدون", "سوى")

QUANTIFIER_TERMS_FR = (
    r"\btous?\b", r"\btoutes?\b", r"\bchaque\b", r"\bplusieurs\b",
    r"\bquelques?\b", r"\baucun(?:e|s)?\b", r"\bcombien\b", r"\bmoins\b",
    r"\bplus\b", r"\d+",
)
QUANTIFIER_TERMS_AR = ("كل ", "جميع", "بعض", "معظم", "عدد", "كم ")

CROSSREF_FR = re.compile(r"\b(?:article|articles|chapitre|chapitres|titre|titres|section)\s+\d+", re.IGNORECASE)
CROSSREF_AR = re.compile(r"(?:المادة|المواد|الفصل|الفصول|الباب|الأبواب)\s*\d+")

QTYPE_FACTUAL_FR = (r"^quand\b", r"^où\b", r"^qui\b", r"^combien\b", r"^en quelle\b")
QTYPE_FACTUAL_AR = ("متى ", "أين ", "من ", "كم ")

QTYPE_DESCRIPTIVE_FR = (r"^quel(?:le|les|s)?\b", r"^qu[''e]est-ce\b", r"^que\b", r"^quoi\b")
QTYPE_DESCRIPTIVE_AR = ("ما ", "ماذا", "ماهي", "ما هي", "ماهو", "ما هو", "اذكر", "عدد ")

QTYPE_REASONING_FR = (r"\bpourquoi\b", r"\bcomment\b", r"\bexpliqu", r"\bjustifi", r"\bquel(?:le|les|s)?\s+rôle\b")
QTYPE_REASONING_AR = ("لماذا", "كيف", "اشرح", "برر", "علل")


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in TOKEN_PATTERN.findall(text)}


def f1_ans_len(a_tokens: int) -> float:
    return min(a_tokens / 500.0, 1.0)


def f2_ctx_len(c_tokens: int) -> float:
    return min(c_tokens / 8000.0, 1.0)


def f3_lexical_overlap(answer: str, context: str) -> float:
    ta, tc = _tokens(answer), _tokens(context)
    if not ta or not tc:
        return 1.0
    jaccard = len(ta & tc) / len(ta | tc)
    return 1.0 - jaccard


def _any_regex(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _any_substr(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def f4_has_negation(question: str, lang: str) -> float:
    q = question or ""
    if lang == "fr":
        return 0.5 if _any_regex(q, NEGATION_TERMS_FR) else 0.0
    return 0.5 if _any_substr(q, NEGATION_TERMS_AR) else 0.0


def f5_has_quantifier(question: str, lang: str) -> float:
    q = question or ""
    if lang == "fr":
        return 0.3 if _any_regex(q, QUANTIFIER_TERMS_FR) else 0.0
    if _any_substr(q, QUANTIFIER_TERMS_AR) or re.search(r"\d+", q):
        return 0.3
    return 0.0


def f6_cross_refs(question: str, context: str, lang: str) -> float:
    text = f"{question} {context}"
    pattern = CROSSREF_FR if lang == "fr" else CROSSREF_AR
    count = len(pattern.findall(text))
    return min(count / 10.0, 1.0)


def f7_question_type(question: str, lang: str) -> float:
    q = (question or "").strip().lower()
    if lang == "fr":
        if _any_regex(q, QTYPE_REASONING_FR):
            return 0.9
        if _any_regex(q, QTYPE_DESCRIPTIVE_FR):
            return 0.5
        if _any_regex(q, QTYPE_FACTUAL_FR):
            return 0.1
        return 0.5
    if _any_substr(q, QTYPE_REASONING_AR):
        return 0.9
    if _any_substr(q, QTYPE_DESCRIPTIVE_AR):
        return 0.5
    if _any_substr(q, QTYPE_FACTUAL_AR):
        return 0.1
    return 0.5


def compute_features(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["f1_ans_len"] = df["a_tokens"].apply(f1_ans_len)
    out["f2_ctx_len"] = df["c_tokens"].apply(f2_ctx_len)
    out["f3_lexical_overlap"] = [
        f3_lexical_overlap(a, c) for a, c in zip(df["answer"], df["context"])
    ]
    out["f4_has_negation"] = df["question"].apply(lambda q: f4_has_negation(q, lang))
    out["f5_has_quantifier"] = df["question"].apply(lambda q: f5_has_quantifier(q, lang))
    out["f6_cross_refs"] = [
        f6_cross_refs(q, c, lang) for q, c in zip(df["question"], df["context"])
    ]
    out["f7_question_type"] = df["question"].apply(lambda q: f7_question_type(q, lang))
    return out


def composite_score(features: pd.DataFrame, weights: dict) -> pd.Series:
    score = pd.Series(0.0, index=features.index)
    for name, w in weights.items():
        score += float(w) * features[name]
    return score.clip(0.0, 1.0)


def assign_difficulty(scores: pd.Series) -> tuple[pd.Series, dict]:
    easy_th = float(np.percentile(scores, 33.33))
    medium_th = float(np.percentile(scores, 66.67))
    labels = np.where(
        scores <= easy_th, "easy",
        np.where(scores <= medium_th, "medium", "hard"),
    )
    return pd.Series(labels, index=scores.index), {"easy": easy_th, "medium": medium_th}


def _decade(doc_date: str) -> str:
    if not doc_date or len(doc_date) < 4 or not doc_date[:4].isdigit():
        return "unknown"
    return f"{doc_date[:3]}0s"


def stratified_split(df: pd.DataFrame, train_frac: float, seed: int) -> pd.Series:
    """Assign 'train'/'test' per row, stratified by (doc_type, decade) inside the df."""
    rng = np.random.default_rng(seed)
    assignments = pd.Series("train", index=df.index)
    strata = df["doc_type"].fillna("") + "||" + df["doc_date"].fillna("").map(_decade)
    for _, group_idx in df.groupby(strata).groups.items():
        idx_array = np.array(list(group_idx))
        if len(idx_array) == 0:
            continue
        rng.shuffle(idx_array)
        n_test = max(1, int(round(len(idx_array) * (1.0 - train_frac)))) if len(idx_array) > 1 else 0
        test_idx = idx_array[:n_test]
        assignments.loc[test_idx] = "test"
    return assignments


def stratify_language(df: pd.DataFrame, lang: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    log.info("[%s] Computing 7 difficulty features on %d rows", lang, len(df))
    feats = compute_features(df, lang)
    scores = composite_score(feats, cfg["stratification_weights"])
    difficulty, thresholds = assign_difficulty(scores)

    df = df.copy()
    df["difficulty_score"] = scores.round(4)
    df["difficulty"] = difficulty

    splits = pd.Series("train", index=df.index)
    for bucket in ("easy", "medium", "hard"):
        mask = df["difficulty"] == bucket
        splits.loc[mask] = stratified_split(
            df.loc[mask], cfg["train_test_split"], cfg["RANDOM_SEED"]
        )
    df["split"] = splits

    counts = df.groupby(["difficulty", "split"]).size().unstack(fill_value=0)
    log.info("[%s] Bucket x split counts:\n%s", lang, counts.to_string())
    log.info(
        "[%s] Thresholds easy≤%.4f medium≤%.4f",
        lang, thresholds["easy"], thresholds["medium"],
    )

    manifest = {
        "easy": df.loc[df["difficulty"] == "easy", "qa_id"].tolist(),
        "medium": df.loc[df["difficulty"] == "medium", "qa_id"].tolist(),
        "hard": df.loc[df["difficulty"] == "hard", "qa_id"].tolist(),
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "train_ids": df.loc[df["split"] == "train", "qa_id"].tolist(),
        "test_ids": df.loc[df["split"] == "test", "qa_id"].tolist(),
    }
    return df, manifest


def _config_hash(cfg: dict) -> str:
    relevant = {
        "stratification_weights": cfg["stratification_weights"],
        "train_test_split": cfg["train_test_split"],
        "RANDOM_SEED": cfg["RANDOM_SEED"],
        "USE_HYBRID_STRATIFICATION": cfg["USE_HYBRID_STRATIFICATION"],
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def main() -> int:
    cfg = load_config()

    qa_fr_path = ROOT / cfg["paths"]["interim_fr_qa"]
    qa_ar_path = ROOT / cfg["paths"]["interim_ar_qa"]
    splits_path = ROOT / cfg["paths"]["interim_splits"]

    log.info("Loading %s", qa_fr_path)
    qa_fr = pd.read_parquet(qa_fr_path)
    log.info("Loading %s", qa_ar_path)
    qa_ar = pd.read_parquet(qa_ar_path)

    qa_fr, manifest_fr = stratify_language(qa_fr, "fr", cfg)
    qa_ar, manifest_ar = stratify_language(qa_ar, "ar", cfg)

    ensure_dir(qa_fr_path.parent)
    qa_fr.to_parquet(qa_fr_path, index=False)
    qa_ar.to_parquet(qa_ar_path, index=False)
    log.info("Updated %s (+difficulty, difficulty_score, split)", qa_fr_path)
    log.info("Updated %s (+difficulty, difficulty_score, split)", qa_ar_path)

    splits = {
        "version": "1.0",
        "date": date.today().isoformat(),
        "seed": cfg["RANDOM_SEED"],
        "config_hash": _config_hash(cfg),
        "fr": manifest_fr,
        "ar": manifest_ar,
    }
    splits_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", splits_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
