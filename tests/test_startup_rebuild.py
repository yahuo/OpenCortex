from __future__ import annotations

import json
from pathlib import Path

import start


def _write_valid_fulltext_index(persist_dir: Path) -> None:
    (persist_dir / "fulltext_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doc_count": 0,
                "avg_chunk_length": 0.0,
                "chunks": [],
                "postings": {},
            }
        ),
        encoding="utf-8",
    )


def _cfg(tmp_path: Path) -> dict[str, str]:
    return {
        "source_dir": str(tmp_path / "docs"),
        "persist_dir": str(tmp_path / "index"),
        "embed_api_key": "embed-key",
        "embed_base_url": "https://embed.example/v1",
        "embed_model": "embed-model",
        "llm_api_key": "llm-key",
        "llm_base_url": "https://llm.example/v1",
        "llm_model": "llm-model",
        "host": "127.0.0.1",
        "port": "8501",
        "api_port": "8502",
    }


def _write_source_doc(cfg: dict[str, str], rel_path: str = "kb/doc.md", content: str = "# Doc\n\nhello") -> Path:
    source_path = Path(cfg["source_dir"]) / rel_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(content, encoding="utf-8")
    return source_path


def test_index_rebuild_state_skips_when_snapshot_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is False
    assert reason == "up_to_date"


def test_index_rebuild_state_detects_source_or_config_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": {"version": 1, "embed_model": "old-model"},
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: {"version": 1, "embed_model": "new-model"})

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "build_config_changed"


def test_index_rebuild_state_requires_existing_index(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    Path(cfg["persist_dir"]).mkdir(parents=True)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "missing_index"


def test_index_rebuild_state_requires_index_pickle_even_when_manifest_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "missing_index"


def test_index_rebuild_state_requires_fulltext_index_even_when_snapshot_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "missing_index"


def test_index_rebuild_state_rejects_invalid_fulltext_index_even_when_snapshot_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    (persist_dir / "fulltext_index.json").write_text("{}", encoding="utf-8")
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "invalid_fulltext_index"


def test_index_rebuild_state_uses_build_config_snapshot_for_fast_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    expected_snapshot = {"version": 1, "config": "same"}
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is False
    assert reason == "up_to_date"


def test_index_rebuild_state_accepts_full_manifest_snapshot_for_fast_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    config_snapshot = {"version": 1, "embed_model": "embed-model", "skip_graph": False}
    manifest_snapshot = {
        **config_snapshot,
        "file_count": 42,
        "source_digest": "abc123",
    }
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": manifest_snapshot,
                "source_snapshot": start.current_source_snapshot(cfg["source_dir"]),
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: config_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is False
    assert reason == "up_to_date"


def test_index_rebuild_state_detects_source_change_from_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    source_path = _write_source_doc(cfg, content="# Doc\n\nold")
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    stored_source_snapshot = start.current_source_snapshot(cfg["source_dir"])
    source_path.write_text("# Doc\n\nnew", encoding="utf-8")
    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": stored_source_snapshot,
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is True
    assert reason == "source_changed"


def test_index_rebuild_state_ignores_unsupported_file_changes_from_dir_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _cfg(tmp_path)
    _write_source_doc(cfg)
    persist_dir = Path(cfg["persist_dir"])
    persist_dir.mkdir(parents=True)
    (persist_dir / "index.faiss").write_text("faiss", encoding="utf-8")
    (persist_dir / "index.pkl").write_text("pickle", encoding="utf-8")
    _write_valid_fulltext_index(persist_dir)
    expected_snapshot = {"version": 1, "embed_model": "embed-model"}
    stored_source_snapshot = start.current_source_snapshot(cfg["source_dir"])

    unsupported_path = Path(cfg["source_dir"]) / "kb" / "image.png"
    unsupported_path.write_bytes(b"png")

    assert (
        start.compare_stored_source_snapshot(cfg["source_dir"], stored_source_snapshot)
        == "dir_mismatch"
    )

    (persist_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "build_snapshot": expected_snapshot,
                "source_snapshot": stored_source_snapshot,
                "fulltext_index_file": "fulltext_index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(start, "build_config_snapshot", lambda *args, **kwargs: expected_snapshot)

    needs_rebuild, reason = start.index_rebuild_state(cfg)

    assert needs_rebuild is False
    assert reason == "up_to_date"
