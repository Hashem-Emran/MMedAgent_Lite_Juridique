"""D2 — Arabic preprocessing for Jurida.

Reads data/raw/{qa.csv, documents.csv}, selects Arabic Q&A (Arabic char ratio ≥
threshold on the question), normalizes Arabic letter forms (alef/ya, optional
ta marbuta), strips tashkeel when configured, cleans HTML remnants, counts
tokens with the shared tokenizer, deduplicates, truncates oversize contexts,
and writes data/interim/{qa_ar.parquet, docs_ar.parquet}.
"""
from __future__ import annotations

import sys

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
    normalize_arabic,
    setup_logger,
)

log = setup_logger("preprocess_ar")


def preprocess(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_qa = ROOT / cfg["paths"]["raw_qa"]
    raw_docs = ROOT / cfg["paths"]["raw_docs"]
    threshold = cfg["LANG_DETECT_THRESHOLD"]
    min_chars = cfg["MIN_QUESTION_CHARS"]
    max_ctx = cfg["MAX_CONTEXT_CHARS"]
    dedup = cfg["DEDUP_QUESTIONS"]
    remove_tashkeel = cfg["REMOVE_TASHKEEL"]
    norm_ta = cfg["NORMALIZE_TA_MARBUTA"]

    log.info("Reading documents metadata: %s", raw_docs)
    docs = pd.read_csv(raw_docs, dtype=str, keep_default_na=False)
    docs["long_title"] = docs["long_title"].astype(str).apply(fix_mojibake)
    docs["doc_type"] = docs["doc_type"].astype(str).apply(fix_mojibake)
    docs["doc_id"] = docs["file_name"].str.replace(r"\.html$", "", regex=True)

    log.info("Reading QA pairs: %s", raw_qa)
    qa = pd.read_csv(raw_qa, dtype=str, keep_default_na=False)
    qa["doc_id"] = qa["file_name"].str.replace(r"\.html$", "", regex=True)

    qa["q_lang"] = qa["Question"].astype(str).apply(
        lambda s: "ar" if arabic_ratio(s) >= threshold else "fr"
    )
    ar = qa[qa["q_lang"] == "ar"].copy()
    log.info("Arabic Q&A: %d / %d", len(ar), len(qa))

    for col in ("Question", "Answer", "Context"):
        ar[col] = (
            ar[col]
            .astype(str)
            .apply(clean_text)
            .apply(lambda t: normalize_arabic(t, remove_tashkeel, norm_ta))
        )

    before_filter = len(ar)
    ar = ar[ar["Question"].str.len() >= min_chars]
    ar = ar[~ar["Answer"].apply(is_placeholder_answer)]
    ar = ar[ar["Context"].str.len() > 0]
    log.info("After quality filter: %d / %d", len(ar), before_filter)

    if dedup:
        ar["_key"] = [dedup_key(q, d) for q, d in zip(ar["Question"], ar["doc_id"])]
        before = len(ar)
        ar = ar.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
        log.info("After deduplication: %d / %d", len(ar), before)

    ar = ar.reset_index(drop=True)
    ar["truncated"] = ar["Context"].str.len() > max_ctx
    ar["Context"] = ar["Context"].str.slice(0, max_ctx)

    log.info("Tokenizing with %s", cfg["tokenizer_name"])
    tok = get_tokenizer(cfg["tokenizer_name"])
    ar["q_tokens"] = [count_tokens(t, tok) for t in ar["Question"]]
    ar["a_tokens"] = [count_tokens(t, tok) for t in ar["Answer"]]
    ar["c_tokens"] = [count_tokens(t, tok) for t in ar["Context"]]

    meta = docs.set_index("doc_id")[["doc_type", "date", "long_title"]].to_dict(orient="index")
    ar["doc_type"] = ar["doc_id"].map(lambda d: meta.get(d, {}).get("doc_type", ""))
    ar["doc_date"] = ar["doc_id"].map(lambda d: meta.get(d, {}).get("date", ""))
    ar["long_title"] = ar["doc_id"].map(lambda d: meta.get(d, {}).get("long_title", ""))

    ar["qa_id"] = [f"ar_{i:05d}" for i in range(len(ar))]
    ar["lang"] = "ar"

    qa_ar = ar.rename(columns={"Question": "question", "Answer": "answer", "Context": "context"})[
        [
            "qa_id", "lang", "question", "answer", "context",
            "doc_id", "doc_type", "doc_date", "long_title",
            "q_tokens", "a_tokens", "c_tokens", "truncated",
        ]
    ]

    docs_ar = docs[docs["doc_id"].isin(qa_ar["doc_id"].unique())][
        ["doc_id", "long_title", "date", "doc_type"]
    ].reset_index(drop=True)

    return qa_ar, docs_ar


def main() -> int:
    cfg = load_config()
    qa_ar, docs_ar = preprocess(cfg)

    out_qa = ROOT / cfg["paths"]["interim_ar_qa"]
    out_docs = ROOT / cfg["paths"]["interim_ar_docs"]
    ensure_dir(out_qa.parent)
    qa_ar.to_parquet(out_qa, index=False)
    docs_ar.to_parquet(out_docs, index=False)

    log.info("Wrote %s (%d rows)", out_qa, len(qa_ar))
    log.info("Wrote %s (%d rows)", out_docs, len(docs_ar))
    log.info(
        "Token stats (median/mean/max) — q:%d/%d/%d a:%d/%d/%d c:%d/%d/%d",
        int(qa_ar["q_tokens"].median()), int(qa_ar["q_tokens"].mean()), int(qa_ar["q_tokens"].max()),
        int(qa_ar["a_tokens"].median()), int(qa_ar["a_tokens"].mean()), int(qa_ar["a_tokens"].max()),
        int(qa_ar["c_tokens"].median()), int(qa_ar["c_tokens"].mean()), int(qa_ar["c_tokens"].max()),
    )
    log.info("Truncated contexts: %d", int(qa_ar["truncated"].sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
