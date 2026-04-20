"""D1 — Copy the Jurida archive files into data/raw/ and record MD5 checksums.

The dataset is already downloaded locally in `archive/`. This script mirrors the
medical `load_dataset("flaviagiammarino/vqa-rad")` step: it produces a verified,
versioned copy in `data/raw/` that downstream scripts consume.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from dataset.jur_utils import ROOT, ensure_dir, load_config, md5_of_file, setup_logger

log = setup_logger("download_jur")


def copy_and_hash(src: Path, dst: Path) -> str:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return md5_of_file(dst)


def main() -> int:
    cfg = load_config()
    src_qa = ROOT / cfg["paths"]["archive_qa"]
    src_docs = ROOT / cfg["paths"]["archive_docs"]
    dst_qa = ROOT / cfg["paths"]["raw_qa"]
    dst_docs = ROOT / cfg["paths"]["raw_docs"]

    missing = [p for p in (src_qa, src_docs) if not p.exists()]
    if missing:
        log.error("Missing archive files: %s", missing)
        return 1

    log.info("Copying %s -> %s", src_qa, dst_qa)
    qa_md5 = copy_and_hash(src_qa, dst_qa)

    log.info("Copying %s -> %s", src_docs, dst_docs)
    docs_md5 = copy_and_hash(src_docs, dst_docs)

    manifest = {
        "qa_csv": {"path": str(dst_qa.relative_to(ROOT)), "md5": qa_md5, "size": dst_qa.stat().st_size},
        "documents_csv": {"path": str(dst_docs.relative_to(ROOT)), "md5": docs_md5, "size": dst_docs.stat().st_size},
    }
    manifest_path = dst_qa.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote manifest %s", manifest_path)
    log.info("qa.csv  md5=%s size=%d", qa_md5, manifest["qa_csv"]["size"])
    log.info("docs.csv md5=%s size=%d", docs_md5, manifest["documents_csv"]["size"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
