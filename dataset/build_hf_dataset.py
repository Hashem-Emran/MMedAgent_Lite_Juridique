"""D2 — Build the final HuggingFace DatasetDict with 12 splits.

Loads enriched data/interim/{qa_fr.parquet, qa_ar.parquet} (already carrying
difficulty, difficulty_score, split columns from stratify_jur.py), concatenates
them, then materializes 12 splits — {fr, ar} × {easy, medium, hard} × {train,
test} — into a HuggingFace DatasetDict saved to data/jurida_processed/.

Downstream teams load with:
    from datasets import load_from_disk
    ds = load_from_disk("data/jurida_processed")
    gp1_train = ds["fr_easy_train"]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from dataset.jur_utils import ROOT, ensure_dir, load_config, setup_logger

log = setup_logger("build_hf_dataset")

FINAL_COLUMNS = [
    "qa_id", "lang", "question", "answer", "context",
    "doc_id", "doc_type", "doc_date", "long_title",
    "difficulty", "difficulty_score",
    "q_tokens", "a_tokens", "c_tokens", "truncated",
    "split",
]


def _subset(df: pd.DataFrame, lang: str, difficulty: str, split: str) -> Dataset:
    mask = (df["lang"] == lang) & (df["difficulty"] == difficulty) & (df["split"] == split)
    subset = df.loc[mask, FINAL_COLUMNS].reset_index(drop=True)
    return Dataset.from_pandas(subset, preserve_index=False)


def build(qa_fr: pd.DataFrame, qa_ar: pd.DataFrame) -> DatasetDict:
    required = {"difficulty", "difficulty_score", "split"}
    for name, df in (("fr", qa_fr), ("ar", qa_ar)):
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"qa_{name} is missing columns {missing}; run stratify_jur.py first")

    combined = pd.concat([qa_fr, qa_ar], ignore_index=True)
    log.info("Combined rows: %d (fr=%d, ar=%d)", len(combined), len(qa_fr), len(qa_ar))

    splits: dict[str, Dataset] = {}
    for lang in ("fr", "ar"):
        for difficulty in ("easy", "medium", "hard"):
            for split in ("train", "test"):
                key = f"{lang}_{difficulty}_{split}"
                ds = _subset(combined, lang, difficulty, split)
                splits[key] = ds
                log.info("%-22s %5d rows", key, len(ds))
    return DatasetDict(splits)


def main() -> int:
    cfg = load_config()
    qa_fr = pd.read_parquet(ROOT / cfg["paths"]["interim_fr_qa"])
    qa_ar = pd.read_parquet(ROOT / cfg["paths"]["interim_ar_qa"])

    ds = build(qa_fr, qa_ar)

    out_dir = ROOT / cfg["paths"]["processed"]
    if out_dir.exists():
        log.info("Clearing existing %s", out_dir)
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    ds.save_to_disk(str(out_dir))
    log.info("Saved DatasetDict to %s (%d splits)", out_dir, len(ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
