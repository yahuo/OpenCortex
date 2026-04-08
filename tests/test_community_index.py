from __future__ import annotations

import json
from pathlib import Path

import pytest

import ragbot


class FakeEmbeddings:
    def __init__(self, *_args, **_kwargs):
        self.dim = 48

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = text.lower().replace("\n", " ").split()
        if not tokens:
            return vector
        for token in tokens:
            slot = hash(token) % self.dim
            vector[slot] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


def _write_community_corpus(root: Path) -> None:
    (root / "工程").mkdir(parents=True)
    (root / "报告").mkdir(parents=True)

    (root / "工程" / "bootstrap_session.py").write_text(
        """
def bootstrap_session(user_id: str) -> dict:
    return {"user_id": user_id, "status": "ready"}
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "session_notes.md").write_text(
        """
# Session Notes

排查 `bootstrap_session()` 的时候，要关注 `orchestration` 阶段。
这里只描述职责，不直接给源码路径。
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "archive_report.py").write_text(
        """
def archive_report(report_id: str) -> dict:
    return {"report_id": report_id, "state": "archived"}
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "report_notes.md").write_text(
        """
# Report Notes

归档报告使用 `archive_report()`，也会经过 `orchestration` 阶段。
这里只描述职责，不直接给源码路径。
""".strip(),
        encoding="utf-8",
    )


def test_build_vectorstore_writes_community_index_with_bridges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    community_index_path = bundle.persist_dir / "community_index.json"
    community_index = json.loads(community_index_path.read_text(encoding="utf-8"))

    assert community_index["community_count"] == 2
    assert (
        community_index["file_to_community"]["工程/session_notes.md"]
        != community_index["file_to_community"]["报告/report_notes.md"]
    )
    assert any(
        bridge["kind"] == "shared_symbol"
        and {bridge["source"], bridge["target"]} == {"工程/session_notes.md", "报告/report_notes.md"}
        for bridge in community_index["bridges"]
    )

    top_symbol_sets = [
        {item["name"] for item in community["top_symbols"]}
        for community in community_index["communities"]
    ]
    assert any("bootstrap_session" in symbols for symbols in top_symbol_sets)
    assert any("archive_report" in symbols for symbols in top_symbol_sets)
