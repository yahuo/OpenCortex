from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import wiki


def _write_manifest_and_normalized_texts(persist_dir: Path) -> dict:
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
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def test_save_query_note_writes_note_and_log(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    manifest = _write_manifest_and_normalized_texts(persist_dir)
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
    _write_manifest_and_normalized_texts(persist_dir)
    created_at = datetime(2026, 4, 9, 10, 30, 0)
    payload = {
        "persist_path": persist_dir,
        "question": "重复问题",
        "answer": "第一次保存的答案。",
        "sources": [
            {
                "source": "工程/bootstrap_session.py",
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


def test_save_query_note_generates_missing_wiki_pages_from_manifest(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    _write_manifest_and_normalized_texts(persist_dir)

    result = wiki.save_query_note(
        persist_path=persist_dir,
        question="旧索引下能否保存？",
        answer="可以，保存前会先补齐 wiki 页面。",
        sources=[
            {
                "source": "工程/bootstrap_session.py",
                "line_start": 1,
                "line_end": 2,
                "snippet": "def bootstrap_session(user_id: str) -> dict:",
            }
        ],
        created_at=datetime(2026, 4, 9, 10, 40, 0),
    )

    assert Path(result["note_path"]).exists()
    assert (persist_dir / "wiki" / "files" / "工程" / "bootstrap_session.py.md").exists()
    note_text = Path(result["note_path"]).read_text(encoding="utf-8")
    assert "../files/工程/bootstrap_session.py.md" in note_text


def test_generate_wiki_preserves_query_note_log_section(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    manifest = _write_manifest_and_normalized_texts(persist_dir)

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


def test_save_query_note_keeps_both_log_entries_under_concurrent_writes(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    _write_manifest_and_normalized_texts(persist_dir)

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def worker(question: str) -> None:
        try:
            barrier.wait()
            wiki.save_query_note(
                persist_path=persist_dir,
                question=question,
                answer=f"{question} 的答案。",
                sources=[
                    {
                        "source": "工程/bootstrap_session.py",
                        "line_start": 1,
                        "line_end": 2,
                        "snippet": "def bootstrap_session(user_id: str) -> dict:",
                    }
                ],
                created_at=datetime(2026, 4, 9, 11, 30, 0),
            )
        except Exception as exc:  # pragma: no cover - test helper
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("并发问题 A",)),
        threading.Thread(target=worker, args=("并发问题 B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    log_text = (persist_dir / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "并发问题 A" in log_text
    assert "并发问题 B" in log_text


def test_generate_wiki_waits_for_shared_write_lock(tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    manifest = _write_manifest_and_normalized_texts(persist_dir)
    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    wiki_dir = persist_dir / "wiki"
    log_path = wiki_dir / "log.md"
    finished = threading.Event()
    errors: list[Exception] = []

    def rebuild() -> None:
        try:
            wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)
        except Exception as exc:  # pragma: no cover - test helper
            errors.append(exc)
        finally:
            finished.set()

    with wiki._write_lock(wiki_dir):
        thread = threading.Thread(target=rebuild)
        thread.start()
        time.sleep(wiki.WRITE_LOCK_POLL_SECONDS * 3)
        assert not finished.is_set()
        wiki._append_query_note_log(
            log_path,
            datetime(2026, 4, 9, 12, 0, 0),
            "锁内追加的问题",
            Path("queries/locked.md"),
        )

    thread.join()

    assert not errors
    assert finished.is_set()
    log_text = log_path.read_text(encoding="utf-8")
    assert "锁内追加的问题" in log_text


def test_write_lock_recovers_from_stale_lock_file(tmp_path: Path) -> None:
    wiki_dir = tmp_path / "index" / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    lock_path = wiki_dir / wiki.WRITE_LOCK_FILENAME
    lock_path.write_text("999999 0\n", encoding="utf-8")
    stale_time = time.time() - (wiki.STALE_LOCK_GRACE_SECONDS + 1)
    os.utime(lock_path, (stale_time, stale_time))

    with wiki._write_lock(wiki_dir):
        assert lock_path.exists()

    assert lock_path.exists()


def test_lock_owner_alive_uses_windows_safe_probe(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / ".write.lock"
    lock_path.write_text("4321 987654321 123\n", encoding="utf-8")
    seen: dict[str, tuple[int, int | None]] = {}

    def fake_windows_probe(pid: int, token: int | None) -> bool:
        seen["args"] = (pid, token)
        return True

    def unexpected_kill(*_args, **_kwargs):
        raise AssertionError("Windows fallback should not call os.kill(pid, 0)")

    monkeypatch.setattr(wiki.os, "name", "nt")
    monkeypatch.setattr(wiki, "_lock_owner_alive_windows", fake_windows_probe)
    monkeypatch.setattr(wiki.os, "kill", unexpected_kill)

    assert wiki._lock_owner_alive(lock_path) is True
    assert seen["args"] == (4321, 987654321)
