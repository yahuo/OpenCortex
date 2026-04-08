from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import wiki


def test_save_query_note_writes_note_and_log(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "bootstrap_session.py.txt").write_text(
        "def bootstrap_session(user_id: str) -> dict:\n    return {'status': 'ready'}\n",
        encoding="utf-8",
    )
    (normalized_dir / "session_notes.md.txt").write_text(
        "# Session Notes\n\n排查 bootstrap_session() 的时候，要关注 orchestration 阶段。\n",
        encoding="utf-8",
    )
    manifest = {
        "build_time": "2026-04-09 10:00:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/bootstrap_session.py",
                "kb": "工程",
                "suffix": ".py",
                "size_kb": 0.1,
                "mtime": "2026-04-09 10:00",
                "chunks": 1,
                "normalized_text": "工程/bootstrap_session.py.txt",
            },
            {
                "name": "工程/session_notes.md",
                "kb": "工程",
                "suffix": ".md",
                "size_kb": 0.1,
                "mtime": "2026-04-09 10:00",
                "chunks": 1,
                "normalized_text": "工程/session_notes.md.txt",
            },
        ],
    }
    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    result = wiki.save_query_note(
        persist_path=persist_dir,
        question="bootstrap_session 的职责是什么？",
        answer="`bootstrap_session()` 负责初始化会话上下文，并返回 ready 状态。",
        sources=[
            {
                "source": "工程/bootstrap_session.py",
                "line_start": 1,
                "line_end": 3,
                "match_kind": "ast",
                "snippet": "def bootstrap_session(user_id: str) -> dict:\n    return {'status': 'ready'}",
            },
            {
                "source": "工程/session_notes.md",
                "time_range": "00:00-00:30",
                "match_kind": "grep",
                "snippet": "排查 bootstrap_session() 的时候，要关注 orchestration 阶段。",
            },
        ],
        created_at=datetime(2026, 4, 9, 10, 30, 0),
    )

    note_path = Path(result["note_path"])
    log_path = Path(result["log_path"])

    assert note_path.exists()
    assert result["note_relpath"].startswith("wiki/queries/2026-04-09-")

    note_text = note_path.read_text(encoding="utf-8")
    assert "# bootstrap_session 的职责是什么？" in note_text
    assert "## 结论" in note_text
    assert "工程/bootstrap_session.py" in note_text
    assert "L1-L3" in note_text
    assert "00:00-00:30" in note_text
    assert "../files/工程/bootstrap_session.py.md" in note_text
    assert "默认不进入最高优先级检索证据链" in note_text

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "## Query Notes" in log_text
    assert "bootstrap_session 的职责是什么？" in log_text
    assert "queries/2026-04-09-" in log_text
    lint_report = json.loads((persist_dir / "lint_report.json").read_text(encoding="utf-8"))
    assert lint_report["summary"]["missing_links"] == 0


def test_save_query_note_generates_unique_path_for_same_day_same_question(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    created_at = datetime(2026, 4, 9, 10, 30, 0)
    payload = {
        "persist_path": persist_dir,
        "question": "重复问题",
        "answer": "第一次保存的答案。",
        "sources": [
            {
                "source": "工程/repeat.md",
                "line_start": 8,
                "line_end": 8,
                "snippet": "repeat snippet",
            }
        ],
        "created_at": created_at,
    }

    first = wiki.save_query_note(**payload)
    second = wiki.save_query_note(**payload)

    assert first["note_path"] != second["note_path"]
    assert Path(first["note_path"]).name == "2026-04-09-重复问题.md"
    assert Path(second["note_path"]).name == "2026-04-09-重复问题-2.md"


def test_generate_wiki_preserves_query_note_log_section(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    normalized_rel = "工程/bootstrap_session.py.txt"
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "bootstrap_session.py.txt").write_text(
        "def bootstrap_session():\n    return {'status': 'ready'}\n",
        encoding="utf-8",
    )
    manifest = {
        "build_time": "2026-04-09 11:00:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/bootstrap_session.py",
                "kb": "工程",
                "suffix": ".py",
                "size_kb": 0.1,
                "mtime": "2026-04-09 11:00",
                "chunks": 1,
                "normalized_text": normalized_rel,
            }
        ],
    }

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)
    wiki.save_query_note(
        persist_path=persist_dir,
        question="bootstrap_session 做什么？",
        answer="负责初始化会话。",
        sources=[
            {
                "source": "工程/bootstrap_session.py",
                "line_start": 1,
                "line_end": 2,
                "snippet": "def bootstrap_session():\n    return {'status': 'ready'}",
            }
        ],
        created_at=datetime(2026, 4, 9, 11, 5, 0),
    )

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    log_text = (persist_dir / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "## Query Notes" in log_text
    assert "bootstrap_session 做什么？" in log_text
