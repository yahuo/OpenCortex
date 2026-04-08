from __future__ import annotations

import json
import os
import time
from pathlib import Path

import wiki


def test_generate_lint_report_detects_stale_orphan_and_missing_links(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_rel = "工程/bootstrap.md.txt"
    normalized_path = persist_dir / "normalized_texts" / normalized_rel
    normalized_path.write_text("# Bootstrap\n\ncontent", encoding="utf-8")

    manifest = {
        "build_time": "2026-04-09 12:00:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/bootstrap.md",
                "kb": "工程",
                "suffix": ".md",
                "size_kb": 0.1,
                "mtime": "2026-04-09 12:00",
                "chunks": 1,
                "normalized_text": normalized_rel,
            }
        ],
    }

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    page_path = persist_dir / "wiki" / "files" / "工程" / "bootstrap.md.md"
    stale_base = time.time() - 60
    os.utime(page_path, (stale_base, stale_base))
    fresh_time = stale_base + 30
    os.utime(normalized_path, (fresh_time, fresh_time))

    orphan_path = persist_dir / "wiki" / "queries" / "orphan.md"
    orphan_path.parent.mkdir(parents=True, exist_ok=True)
    orphan_path.write_text(
        "# Orphan\n\n[broken](../missing.md)\n",
        encoding="utf-8",
    )

    report = wiki.generate_lint_report(persist_path=persist_dir, manifest=manifest)

    assert report["summary"] == {
        "stale_pages": 1,
        "orphan_pages": 1,
        "missing_links": 1,
    }
    assert report["stale_pages"][0]["page"] == "files/工程/bootstrap.md.md"
    assert report["orphan_pages"][0]["page"] == "queries/orphan.md"
    assert report["missing_links"][0]["page"] == "queries/orphan.md"
    assert report["missing_links"][0]["target"] == "../missing.md"

    saved = json.loads((persist_dir / "lint_report.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]
