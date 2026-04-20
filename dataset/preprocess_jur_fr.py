"""D1 — French preprocessing for Jurida.

Reads data/raw/{qa.csv, documents.csv}, selects French Q&A (Arabic char ratio <
threshold), repairs cp1252 mojibake on documents metadata, cleans HTML remnants,
counts tokens with the shared tokenizer, deduplicates, truncates oversize
contexts, and writes data/interim/{qa_fr.parquet, docs_fr.parquet}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from dataset.jur_utils import (
    ROOT,
    arabic_ratio,
    clean_text,
    count_tokens,
    dedup_key,
    ensure_dir,
    fix_mojibake,
    get_tokenizer,
    is_placeholder_answer,
    load_config,
    normalize_fr,
    setup_logger,
)

log = setup_logger("preprocess_fr")


def preprocess(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_qa = ROOT / cfg["paths"]["raw_qa"]
    raw_docs = ROOT / cfg["paths"]["raw_docs"]
    threshold = cfg["LANG_DETECT_THRESHOLD"]
    min_chars = cfg["MIN_QUESTION_CHARS"]
    max_ctx = cfg["MAX_CONTEXT_CHARS"]
    dedup = cfg["DEDUP_QUESTIONS"]

    log.info("Reading documents metadata: %s", raw_docs)
    docs = pd.read_csv(raw_docs, dtype=str, keep_default_na=False)
    docs["file_content"] = docs["file_content"].astype(str).apply(fix_mojibake)
    docs["long_title"] = docs["long_title"].astype(str).apply(fix_mojibake)
    docs["doc_type"] = docs["doc_type"].astype(str).apply(fix_mojibake)
    docs["doc_id"] = docs["file_name"].str.replace(r"\.html$", "", regex=True)

    log.info("Reading QA pairs: %s", raw_qa)
    qa = pd.read_csv(raw_qa, dtype=str, keep_default_na=False)
    qa["doc_id"] = qa["file_name"].str.replace(r"\.html$", "", regex=True)

    qa["q_lang"] = qa["Question"].astype(str).apply(
        lambda s: "ar" if arabic_ratio(s) >= threshold else "fr"
    )
    fr = qa[qa["q_lang"] == "fr"].copy()
    log.info("French Q&A: %d / %d", len(fr), len(qa))

    for col in ("Question", "Answer", "Context"):
        fr[col] = fr[col].astype(str).apply(fix_mojibake).apply(clean_text).apply(normalize_fr)

    before_filter = len(fr)
    fr = fr[fr["Question"].str.len() >= min_chars]
    fr = fr[~fr["Answer"].apply(is_placeholder_answer)]
    fr = fr[fr["Context"].str.len() > 0]
    log.info("After quality filter: %d / %d", len(fr), before_filter)

    if dedup:
        fr["_key"] = [dedup_key(q, d) for q, d in zip(fr["Question"], fr["doc_id"])]
        before = len(fr)
        fr = fr.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
        log.info("After deduplication: %d / %d", len(fr), before)

    fr = fr.reset_index(drop=True)
    fr["truncated"] = fr["Context"].str.len() > max_ctx
    fr["Context"] = fr["Context"].str.slice(0, max_ctx)

    log.info("Tokenizing with %s", cfg["tokenizer_name"])
    tok = get_tokenizer(cfg["tokenizer_name"])
    fr["q_tokens"] = [count_tokens(t, tok) for t in fr["Question"]]
    fr["a_tokens"] = [count_tokens(t, tok) for t in fr["Answer"]]
    fr["c_tokens"] = [count_tokens(t, tok) for t in fr["Context"]]

    meta = docs.set_index("doc_id")[["doc_type", "date", "long_title"]].to_dict(orient="index")
    fr["doc_type"] = fr["doc_id"].map(lambda d: meta.get(d, {}).get("doc_type", ""))
    fr["doc_date"] = fr["doc_id"].map(lambda d: meta.get(d, {}).get("date", ""))
    fr["long_title"] = fr["doc_id"].map(lambda d: meta.get(d, {}).get("long_title", ""))

    fr["qa_id"] = [f"fr_{i:05d}" for i in range(len(fr))]
    fr["lang"] = "fr"

    qa_fr = fr.rename(columns={"Question": "question", "Answer": "answer", "Context": "context"})[
        [
            "qa_id", "lang", "question", "answer", "context",
            "doc_id", "doc_type", "doc_date", "long_title",
            "q_tokens", "a_tokens", "c_tokens", "truncated",
        ]
    ]

    docs_fr = docs[docs["doc_id"].isin(qa_fr["doc_id"].unique())][
        ["doc_id", "long_title", "date", "doc_type", "file_content"]
    ].reset_index(drop=True)

    return qa_fr, docs_fr


def main() -> int:
    cfg = load_config()
    qa_fr, docs_fr = preprocess(cfg)

    out_qa = ROOT / cfg["paths"]["interim_fr_qa"]
    out_docs = ROOT / cfg["paths"]["interim_fr_docs"]
    ensure_dir(out_qa.parent)
    qa_fr.to_parquet(out_qa, index=False)
    docs_fr.to_parquet(out_docs, index=False)

    log.info("Wrote %s (%d rows)", out_qa, len(qa_fr))
    log.info("Wrote %s (%d rows)", out_docs, len(docs_fr))
    log.info(
        "Token stats (median/mean/max) — q:%d/%d/%d a:%d/%d/%d c:%d/%d/%d",
        int(qa_fr["q_tokens"].median()), int(qa_fr["q_tokens"].mean()), int(qa_fr["q_tokens"].max()),
        int(qa_fr["a_tokens"].median()), int(qa_fr["a_tokens"].mean()), int(qa_fr["a_tokens"].max()),
        int(qa_fr["c_tokens"].median()), int(qa_fr["c_tokens"].mean()), int(qa_fr["c_tokens"].max()),
    )
    log.info("Truncated contexts: %d", int(qa_fr["truncated"].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
