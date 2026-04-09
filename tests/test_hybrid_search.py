from __future__ import annotations

import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

import ragbot
import wiki


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


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, *_args, **kwargs):
        self.temperature = kwargs.get("temperature", 0.0)

    def invoke(self, _prompt: str):
        return FakeMessage(
            json.dumps(
                {
                    "symbols": ["bootstrap_session"],
                    "keywords": ["bootstrap_session", "session bootstrap"],
                    "path_globs": ["*session*"],
                    "semantic_query": "session bootstrap flow",
                    "reason": "focus on bootstrap symbol",
                },
                ensure_ascii=False,
            )
        )

    def stream(self, _prompt: str):
        yield FakeMessage("mock answer")


def _write_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "工程").mkdir(parents=True)
    (root / "工程" / "nested").mkdir(parents=True)
    (root / "node_modules").mkdir(parents=True)

    (root / "产品" / "settings.yaml").write_text(
        """
service:
  api_key: cortex-secret
  port: 8502
search:
  mode: hybrid
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "bootstrap_session.py").write_text(
        '''
import json


def bootstrap_session(user_id: str) -> dict:
    """Concrete initialization steps for the user session bootstrap flow."""
    payload = {"user_id": user_id, "status": "ready"}
    return payload


class SessionCoordinator:
    def create(self, user_id: str) -> dict:
        return bootstrap_session(user_id)
'''.strip(),
        encoding="utf-8",
    )
    (root / "工程" / "startup.md").write_text(
        """
# Startup

The startup workflow documents the boot sequence and startup orchestration.
It explains how the assistant loads configuration and prepares the session.
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "session_playbook.md").write_text(
        """
# Session Playbook

The implementation details are documented in [bootstrap module](./bootstrap_session.py).
Use `bootstrap_session()` when preparing the concrete initialization steps.
""".strip(),
        encoding="utf-8",
    )
    (root / "产品" / "session_bridge.md").write_text(
        """
# Session Bridge

This product note references ../工程/bootstrap_session.py for internal background only.
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "nested" / "invalid.py").write_text(
        "def broken(:\n    pass\n",
        encoding="utf-8",
    )
    (root / "产品" / "manual.pdf").write_text("placeholder", encoding="utf-8")
    (root / "node_modules" / "ignored.py").write_text("IGNORED = True\n", encoding="utf-8")


@pytest.fixture()
def search_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "_convert_binary_to_markdown",
        lambda _path: "# PDF Manual\n\nsession bootstrap manual and startup references",
    )

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )
    assert bundle is not None
    return bundle


def test_build_vectorstore_writes_search_artifacts(search_bundle):
    cache_path = search_bundle.cache_path_for("产品/settings.yaml")
    assert cache_path is not None and cache_path.exists()
    cache_text = cache_path.read_text(encoding="utf-8")
    assert "service.api_key = cortex-secret" in cache_text
    assert "node_modules/ignored.py" not in search_bundle.files_by_source
    assert any(record["name"] == "bootstrap_session" for record in search_bundle.symbol_index)
    graph_path = search_bundle.persist_dir / "document_graph.json"
    assert graph_path.exists()
    assert any(
        edge["target"] == "工程/bootstrap_session.py"
        for edge in search_bundle.graph_neighbors["工程/session_playbook.md"]
    )
    entity_graph_path = search_bundle.persist_dir / "entity_graph.json"
    assert entity_graph_path.exists()
    entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))
    assert search_bundle.manifest["entity_graph_file"] == "entity_graph.json"
    assert search_bundle.manifest["semantic_extract_cache_file"] == "semantic_extract_cache.json"
    assert search_bundle.manifest["semantic_graph_stats"]["enabled"] is False
    assert search_bundle.entity_graph["node_count"] == len(search_bundle.entity_graph["nodes"])
    assert "file:工程/bootstrap_session.py" in search_bundle.entity_nodes_by_id
    assert any(
        edge["target"] == "symbol:工程/bootstrap_session.py:function:bootstrap_session:4"
        and edge["type"] == "references"
        for edge in search_bundle.entity_edges_by_source["section:工程/session_playbook.md:0"]
    )
    assert any(
        node["type"] == "file" and node["source"] == "工程/bootstrap_session.py"
        for node in entity_graph["nodes"]
    )
    assert any(
        node["type"] == "section"
        and node["source"] == "工程/session_playbook.md"
        and node["name"] == "Session Playbook"
        for node in entity_graph["nodes"]
    )
    assert any(
        node["type"] == "symbol"
        and node["source"] == "工程/bootstrap_session.py"
        and node["qualified_name"] == "bootstrap_session"
        for node in entity_graph["nodes"]
    )
    assert any(
        edge["source"] == "section:工程/session_playbook.md:0"
        and edge["target"] == "symbol:工程/bootstrap_session.py:function:bootstrap_session:4"
        and edge["type"] == "references"
        for edge in entity_graph["edges"]
    )
    community_index_path = search_bundle.persist_dir / "community_index.json"
    assert community_index_path.exists()
    community_index = json.loads(community_index_path.read_text(encoding="utf-8"))
    assert search_bundle.manifest["community_index_file"] == "community_index.json"
    assert community_index["community_count"] >= 1
    assert community_index["file_to_community"]["工程/session_playbook.md"].startswith("community-")
    graph_report_path = search_bundle.persist_dir / "reports" / "GRAPH_REPORT.md"
    assert graph_report_path.exists()
    graph_report = graph_report_path.read_text(encoding="utf-8")
    assert search_bundle.manifest["graph_report_file"] == "reports/GRAPH_REPORT.md"
    assert "# 图谱报告" in graph_report
    lint_report_path = search_bundle.persist_dir / "lint_report.json"
    assert lint_report_path.exists()
    lint_report = json.loads(lint_report_path.read_text(encoding="utf-8"))
    assert search_bundle.manifest["lint_report_file"] == "lint_report.json"
    assert lint_report["summary"] == {
        "stale_pages": 0,
        "orphan_pages": 0,
        "missing_links": 0,
    }
    wiki_index_path = search_bundle.persist_dir / "wiki" / "index.md"
    assert wiki_index_path.exists()
    wiki_index = wiki_index_path.read_text(encoding="utf-8")
    assert "[工程/session_playbook.md](files/工程/session_playbook.md.md)" in wiki_index
    wiki_file_path = search_bundle.persist_dir / "wiki" / "files" / "工程" / "session_playbook.md.md"
    assert wiki_file_path.exists()
    wiki_file = wiki_file_path.read_text(encoding="utf-8")
    assert "- 知识库：`工程`" in wiki_file
    assert "bootstrap_session.py" in wiki_file
    assert (search_bundle.persist_dir / "wiki" / "log.md").exists()
    assert any(page["kind"] == "index" for page in search_bundle.wiki_pages)
    assert any(
        page["kind"] == "file" and page["relpath"] == "files/工程/session_playbook.md.md"
        for page in search_bundle.wiki_pages
    )
    assert any(page["kind"] == "community" for page in search_bundle.wiki_pages)


def test_build_vectorstore_uses_faster_embedding_defaults(tmp_path: Path, monkeypatch) -> None:
    docs = [
        ragbot.Document(
            page_content=f"chunk {index}",
            metadata={"source": "工程/bootstrap_session.py", "chunk_index": index},
        )
        for index in range(ragbot.DEFAULT_EMBED_BATCH_SIZE + 1)
    ]
    batch_sizes: list[int] = []
    sleep_calls: list[float] = []

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, batch, _embeddings):
            batch_sizes.append(len(batch))
            return cls()

        def add_documents(self, batch):
            batch_sizes.append(len(batch))

        def save_local(self, _path: str) -> None:
            return None

    fake_bundle = SimpleNamespace(manifest={})

    monkeypatch.delenv("EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("EMBED_BATCH_SLEEP_SECONDS", raising=False)
    monkeypatch.setattr(ragbot, "_build_documents", lambda _source_dir: (docs, []))
    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "FAISS", FakeVectorStore)
    monkeypatch.setattr(
        ragbot,
        "_write_index_artifacts",
        lambda **_kwargs: {"files": [], "normalized_text_dir": ragbot.NORMALIZED_TEXT_DIRNAME},
    )
    monkeypatch.setattr(ragbot, "load_search_bundle", lambda **_kwargs: fake_bundle)
    monkeypatch.setattr(ragbot.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(wiki, "generate_wiki", lambda **_kwargs: {"pages": 0, "lint_issues": 0})

    ragbot.build_vectorstore(
        md_dir=str(tmp_path / "docs"),
        embed_api_key="fake-key",
        persist_dir=str(tmp_path / "index"),
    )

    assert batch_sizes == [ragbot.DEFAULT_EMBED_BATCH_SIZE, 1]
    assert sleep_calls == []


def test_build_vectorstore_still_honors_configured_batch_sleep(tmp_path: Path, monkeypatch) -> None:
    docs = [
        ragbot.Document(
            page_content=f"chunk {index}",
            metadata={"source": "工程/bootstrap_session.py", "chunk_index": index},
        )
        for index in range(3)
    ]
    sleep_calls: list[float] = []

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, batch, _embeddings):
            return cls()

        def add_documents(self, batch):
            return None

        def save_local(self, _path: str) -> None:
            return None

    fake_bundle = SimpleNamespace(manifest={})

    monkeypatch.setenv("EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("EMBED_BATCH_SLEEP_SECONDS", "0.25")
    monkeypatch.setattr(ragbot, "_build_documents", lambda _source_dir: (docs, []))
    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "FAISS", FakeVectorStore)
    monkeypatch.setattr(
        ragbot,
        "_write_index_artifacts",
        lambda **_kwargs: {"files": [], "normalized_text_dir": ragbot.NORMALIZED_TEXT_DIRNAME},
    )
    monkeypatch.setattr(ragbot, "load_search_bundle", lambda **_kwargs: fake_bundle)
    monkeypatch.setattr(ragbot.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(wiki, "generate_wiki", lambda **_kwargs: {"pages": 0, "lint_issues": 0})

    ragbot.build_vectorstore(
        md_dir=str(tmp_path / "docs"),
        embed_api_key="fake-key",
        persist_dir=str(tmp_path / "index"),
    )

    assert sleep_calls == [0.25, 0.25]


def test_build_vectorstore_can_skip_graph_semantic_and_wiki(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "_convert_binary_to_markdown",
        lambda _path: "# PDF Manual\n\nsession bootstrap manual and startup references",
    )
    monkeypatch.setattr(
        ragbot,
        "_build_document_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip document graph")),
    )
    monkeypatch.setattr(
        ragbot,
        "_extract_semantic_sections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should skip semantic extraction")),
    )
    monkeypatch.setattr(
        wiki,
        "generate_wiki",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should skip wiki generation")),
    )
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    assert bundle.manifest["build_flags"] == {
        "skip_graph": True,
        "skip_semantic": True,
        "skip_wiki": True,
    }
    assert bundle.document_graph["edge_count"] == 0
    assert all(node["type"] == "file" for node in bundle.entity_graph["nodes"])
    assert bundle.manifest["semantic_graph_stats"]["reason"] == "skipped_by_graph"
    assert bundle.wiki_pages == []
    assert not (index_dir / "wiki" / "index.md").exists()


def test_build_vectorstore_can_embed_batches_concurrently(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    for idx in range(3):
        (docs_dir / f"doc-{idx}.md").write_text(
            f"# Doc {idx}\n\nchunk {idx}\n",
            encoding="utf-8",
        )

    tracker = {
        "active": 0,
        "max_active": 0,
        "lock": threading.Lock(),
        "overlap_ready": threading.Event(),
    }

    class BlockingEmbeddings(FakeEmbeddings):
        def __init__(self, shared_tracker: dict[str, object]):
            super().__init__()
            self.shared_tracker = shared_tracker

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            lock = self.shared_tracker["lock"]
            with lock:
                self.shared_tracker["active"] += 1
                self.shared_tracker["max_active"] = max(
                    self.shared_tracker["max_active"],
                    self.shared_tracker["active"],
                )
                if self.shared_tracker["active"] >= 2:
                    self.shared_tracker["overlap_ready"].set()
            self.shared_tracker["overlap_ready"].wait(timeout=0.5)
            try:
                return super().embed_documents(texts)
            finally:
                with lock:
                    self.shared_tracker["active"] -= 1

    monkeypatch.setattr(
        ragbot,
        "make_embeddings",
        lambda *args, **kwargs: BlockingEmbeddings(tracker),
    )
    monkeypatch.setenv("EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    assert tracker["max_active"] >= 2


def test_build_vectorstore_surfaces_rate_limit_progress_in_concurrent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    for idx in range(2):
        (docs_dir / f"doc-{idx}.md").write_text(
            f"# Doc {idx}\n\nchunk {idx}\n",
            encoding="utf-8",
        )

    progress_messages: list[str] = []

    class RetryOnceEmbeddings(FakeEmbeddings):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 rate limit")
            return super().embed_documents(texts)

    monkeypatch.setattr(
        ragbot,
        "make_embeddings",
        lambda *args, **kwargs: RetryOnceEmbeddings(),
    )
    monkeypatch.setattr(ragbot.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        progress_callback=lambda _current, _total, message: progress_messages.append(message),
    )

    assert any("触发 embedding 限流" in message for message in progress_messages)


def test_build_vectorstore_ignores_batch_sleep_in_concurrent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    for idx in range(2):
        (docs_dir / f"doc-{idx}.md").write_text(
            f"# Doc {idx}\n\nchunk {idx}\n",
            encoding="utf-8",
        )

    sleep_calls: list[float] = []
    progress_messages: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setenv("EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("EMBED_BATCH_SLEEP_SECONDS", "0.25")
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        progress_callback=lambda _current, _total, message: progress_messages.append(message),
    )

    assert sleep_calls == []
    assert any("并发模式下已忽略" in message for message in progress_messages)


def test_build_vectorstore_surfaces_worker_failure_in_concurrent_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    for idx in range(3):
        (docs_dir / f"doc-{idx}.md").write_text(
            f"# Doc {idx}\n\nchunk content {idx}\n",
            encoding="utf-8",
        )

    class FailingEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if any("chunk content 1" in text for text in texts):
                raise RuntimeError("simulated upstream outage")
            return super().embed_documents(texts)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FailingEmbeddings())
    monkeypatch.setattr(ragbot.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("EMBED_BATCH_SIZE", "1")
    monkeypatch.setenv("EMBED_CONCURRENCY", "2")
    monkeypatch.setenv("EMBED_MAX_RETRIES", "0")
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    with pytest.raises(RuntimeError, match="simulated upstream outage"):
        ragbot.build_vectorstore(
            md_dir=str(docs_dir),
            embed_api_key="fake-key",
            persist_dir=str(index_dir),
        )

    # 非限流异常必须把构建失败传给调用者，且不允许残留任何已落盘的 FAISS artifact。
    assert not (index_dir / "index.faiss").exists()
    assert not (index_dir / "index.pkl").exists()
    assert not (index_dir / "index_manifest.json").exists()
    # 临时 staged cache 目录由 TemporaryDirectory 在异常路径上清理，persist_dir 下不应残留。
    assert not (index_dir / ragbot.NORMALIZED_TEXT_DIRNAME).exists()
    leftover_staging = [p for p in index_dir.glob(".opencortex-build-*")]
    assert leftover_staging == []


def test_index_source_files_spills_normalized_text_to_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    source_path = docs_dir / "notes.md"
    source_path.write_text(
        """
# Notes

The bootstrap session should load config and then start the assistant.
""".strip(),
        encoding="utf-8",
    )

    cache_root = tmp_path / "build-cache"
    indexed_files, total_chunks = ragbot._index_source_files(str(docs_dir), cache_root)

    assert total_chunks == 1
    assert len(indexed_files) == 1

    indexed = indexed_files[0]
    assert indexed.normalized_text is None
    assert indexed.chunks is None
    assert indexed.normalized_text_path is not None and indexed.normalized_text_path.exists()
    assert indexed.chunk_count == 1

    monkeypatch.setattr(
        ragbot.IndexedFile,
        "get_normalized_text",
        lambda _self: (_ for _ in ()).throw(AssertionError("should not materialize the cached file")),
    )
    documents = list(ragbot._iter_documents_for_indexed_files(indexed_files))
    assert len(documents) == 1
    assert documents[0].metadata["source"] == "notes.md"
    assert "bootstrap session" in documents[0].page_content


def test_split_lines_into_chunks_uses_env_chunk_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "12")
    monkeypatch.setenv("CHUNK_OVERLAP_LINES", "1")

    chunks = ragbot._split_lines_into_chunks(["aaaaa", "bbbbb", "ccccc"])

    assert [(chunk.line_start, chunk.line_end) for chunk in chunks] == [(1, 2), (2, 3)]


def test_parse_wechat_messages_filters_export_noise() -> None:
    content = """
# 聊天记录: Demo

> 导出时间: 2026-04-10 10:00:00

---

**系统提示** `[2024-01-01 10:00:00]`
*]]></plain><br><template><![CDATA["$username$"邀请你加入了群聊]]></template><br><link_list>

**张三** `[2024-01-01 10:01:00]`
![图片](../images/demo.png)

**李四** `[2024-01-01 10:02:00]`
📸 `[加密高级图片: 微信 V2 协议封锁，请前往电脑端原生查看]`

**王五** `[2024-01-01 10:03:00]`
真实进展：服务已经恢复
""".strip()

    messages = ragbot._parse_md_messages_text(content)

    assert len(messages) == 1
    assert messages[0]["sender"] == "王五"
    assert messages[0]["content"] == "真实进展：服务已经恢复"


def test_parse_wechat_messages_drops_system_pattern_without_xml() -> None:
    messages = ragbot._parse_md_messages_text(
        """
**系统提示** `[2024-01-01 10:00:00]`
张三 撤回了一条消息
""".strip()
    )

    assert messages == []


def test_iter_chunks_from_cached_file_respects_max_chunks_per_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_CHUNKS_PER_FILE", "2")
    path = tmp_path / "wechat.md"
    path.write_text(
        """
**张三** `[2024-01-01 10:00:00]`
第一段

**张三** `[2024-01-01 11:00:00]`
第二段

**张三** `[2024-01-01 12:00:00]`
第三段
""".strip(),
        encoding="utf-8",
    )

    chunks = list(ragbot._iter_chunks_from_cached_file(path, ".md", "wechat_markdown"))

    assert len(chunks) == 2
    assert "第一段" not in chunks[0].text
    assert "第二段" in chunks[0].text
    assert "第三段" in chunks[1].text


def test_iter_chunks_from_cached_file_filters_wechat_image_noise(tmp_path: Path) -> None:
    path = tmp_path / "wechat-noise.md"
    path.write_text(
        """
**张三** `[2024-01-01 10:00:00]`
![图片](../images/demo.png)
真实内容
""".strip(),
        encoding="utf-8",
    )

    chunks = list(ragbot._iter_chunks_from_cached_file(path, ".md", "wechat_markdown"))

    assert len(chunks) == 1
    assert ".png" not in chunks[0].text
    assert "真实内容" in chunks[0].text


def test_build_vectorstore_records_truncated_files_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    progress_messages: list[str] = []
    (docs_dir / "wechat.md").write_text(
        """
**张三** `[2024-01-01 10:00:00]`
第一段

**张三** `[2024-01-01 11:00:00]`
第二段

**张三** `[2024-01-01 12:00:00]`
第三段
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setenv("MAX_CHUNKS_PER_FILE", "2")
    monkeypatch.setenv("SKIP_GRAPH", "1")
    monkeypatch.setenv("SKIP_SEMANTIC", "1")
    monkeypatch.setenv("SKIP_WIKI", "1")

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        progress_callback=lambda _current, _total, message: progress_messages.append(message),
    )

    file_entry = bundle.manifest["files"][0]
    assert file_entry["chunks"] == 2
    assert file_entry["original_chunks"] == 3
    assert file_entry["truncated"] is True
    assert any("MAX_CHUNKS_PER_FILE 被截断" in message for message in progress_messages)


def test_markdown_heading_chunker_yields_before_consuming_entire_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    long_line = "x" * 400

    def fake_iter_text_file_lines(_path: Path):
        calls["count"] += 1
        yield "# Huge Section"
        for index in range(20):
            if calls["count"] == 2 and index >= 8:
                raise AssertionError("should yield the first chunk before reading the full section")
            yield long_line

    monkeypatch.setattr(ragbot, "_iter_text_file_lines", fake_iter_text_file_lines)

    chunk_iter = ragbot._iter_markdown_heading_chunks_from_file(tmp_path / "ignored.md")
    first_chunk = next(chunk_iter)

    assert first_chunk.label == "Huge Section"
    assert first_chunk.line_start == 1
    assert first_chunk.line_end < 10


def test_markdown_heading_chunker_reads_no_heading_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_iter_text_file_lines(_path: Path):
        calls["count"] += 1
        yield "plain intro"
        yield "plain body"

    monkeypatch.setattr(ragbot, "_iter_text_file_lines", fake_iter_text_file_lines)

    chunks = list(ragbot._iter_markdown_heading_chunks_from_file(Path("ignored.md")))

    assert calls["count"] == 1
    assert len(chunks) == 1
    assert chunks[0].label == "preface"


def test_build_vectorstore_stages_cache_under_persist_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    index_dir = tmp_path / "index"
    sentinel = SimpleNamespace(manifest={})
    seen: dict[str, Path] = {}

    def fake_index_source_files(_source_dir: str, cache_root: Path) -> tuple[list[ragbot.IndexedFile], int]:
        seen["cache_root"] = cache_root
        return [], 0

    def fake_build_vectorstore_from_document_stream(**kwargs):
        seen["staged_normalized_dir"] = kwargs["staged_normalized_dir"]
        return sentinel

    monkeypatch.setattr(ragbot, "_index_source_files", fake_index_source_files)
    monkeypatch.setattr(ragbot, "_build_vectorstore_from_document_stream", fake_build_vectorstore_from_document_stream)

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    assert index_dir in seen["cache_root"].parents
    assert seen["staged_normalized_dir"] == seen["cache_root"]


def test_load_search_bundle_normalizes_entity_graph(search_bundle, monkeypatch: pytest.MonkeyPatch):
    entity_graph_path = search_bundle.persist_dir / "entity_graph.json"
    entity_graph = json.loads(entity_graph_path.read_text(encoding="utf-8"))

    def rewrite_node_id(node_id: str) -> str:
        return node_id.replace("/", "\\")

    for node in entity_graph["nodes"]:
        if isinstance(node.get("id"), str):
            node["id"] = rewrite_node_id(node["id"])
        for key in ("source", "file", "path"):
            value = node.get(key)
            if isinstance(value, str):
                node[key] = value.replace("/", "\\")

    duplicated_edge = None
    for edge in entity_graph["edges"]:
        if isinstance(edge.get("source"), str):
            edge["source"] = rewrite_node_id(edge["source"])
        if isinstance(edge.get("target"), str):
            edge["target"] = rewrite_node_id(edge["target"])
        evidence = edge.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("source"), str):
            evidence["source"] = evidence["source"].replace("/", "\\")
        if (
            duplicated_edge is None
            and edge.get("source") == "section:工程\\session_playbook.md:0"
            and edge.get("target") == "symbol:工程\\bootstrap_session.py:function:bootstrap_session:4"
            and edge.get("type") == "references"
        ):
            duplicated_edge = json.loads(json.dumps(edge, ensure_ascii=False))

    assert duplicated_edge is not None
    entity_graph["edges"].append(duplicated_edge)
    entity_graph_path.write_text(json.dumps(entity_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    reloaded = ragbot.load_search_bundle(
        embed_api_key="fake-key",
        persist_dir=str(search_bundle.persist_dir),
    )

    assert reloaded is not None
    assert "file:工程/bootstrap_session.py" in reloaded.entity_nodes_by_id
    assert all("\\" not in node["id"] for node in reloaded.entity_graph["nodes"])
    assert all("\\" not in node.get("source", "") for node in reloaded.entity_graph["nodes"])
    section_edges = reloaded.entity_edges_by_source["section:工程/session_playbook.md:0"]
    assert sum(
        1
        for edge in section_edges
        if edge["target"] == "symbol:工程/bootstrap_session.py:function:bootstrap_session:4"
        and edge["type"] == "references"
    ) == 1
    assert all("\\" not in edge["source"] for edge in reloaded.entity_graph["edges"])
    assert all("\\" not in edge["target"] for edge in reloaded.entity_graph["edges"])


def test_generate_wiki_escapes_markdown_link_metacharacters(tmp_path: Path):
    persist_dir = tmp_path / "index"
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_rel = "工程/plan]v2#draft.md.txt"
    (persist_dir / "normalized_texts" / normalized_rel).write_text(
        "# Draft\n\nbootstrap session draft notes",
        encoding="utf-8",
    )

    manifest = {
        "build_time": "2026-04-08 16:00:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/plan]v2#draft.md",
                "kb": "工程",
                "suffix": ".md",
                "size_kb": 0.1,
                "mtime": "2026-04-08 16:00",
                "chunks": 1,
                "normalized_text": normalized_rel,
            }
        ],
    }

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    index_text = (persist_dir / "wiki" / "index.md").read_text(encoding="utf-8")
    assert (
        r"[工程/plan\]v2#draft.md](files/工程/plan]v2%23draft.md.md)"
        in index_text
    )
    assert (
        persist_dir / "wiki" / "files" / "工程" / "plan]v2#draft.md.md"
    ).exists()


def test_generate_wiki_wraps_preview_with_safe_outer_fence(tmp_path: Path):
    persist_dir = tmp_path / "index"
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_rel = "工程/fenced.md.txt"
    (persist_dir / "normalized_texts" / normalized_rel).write_text(
        "# Example\n\n```python\nprint('hi')\n```",
        encoding="utf-8",
    )

    manifest = {
        "build_time": "2026-04-08 16:30:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/fenced.md",
                "kb": "工程",
                "suffix": ".md",
                "size_kb": 0.1,
                "mtime": "2026-04-08 16:30",
                "chunks": 1,
                "normalized_text": normalized_rel,
            }
        ],
    }

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    page_text = (persist_dir / "wiki" / "files" / "工程" / "fenced.md.md").read_text(
        encoding="utf-8"
    )
    assert "\n~~~text\n" in page_text
    assert "```python" in page_text
    assert "\n~~~\n\n## 备注\n" in page_text


def test_generate_wiki_escapes_backticks_in_file_page_metadata(tmp_path: Path):
    persist_dir = tmp_path / "index"
    normalized_dir = persist_dir / "normalized_texts" / "工程"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_rel = "工程/foo`bar.md.txt"
    (persist_dir / "normalized_texts" / normalized_rel).write_text(
        "# Title\n\ncontent",
        encoding="utf-8",
    )

    manifest = {
        "build_time": "2026-04-08 17:00:00",
        "normalized_text_dir": "normalized_texts",
        "files": [
            {
                "name": "工程/foo`bar.md",
                "kb": "工程",
                "suffix": ".md",
                "size_kb": 0.1,
                "mtime": "2026-04-08 17:00",
                "chunks": 1,
                "normalized_text": normalized_rel,
            }
        ],
    }

    wiki.generate_wiki(persist_path=persist_dir, manifest=manifest)

    page_text = (persist_dir / "wiki" / "files" / "工程" / "foo`bar.md.md").read_text(
        encoding="utf-8"
    )
    assert "- 标准化文本：``工程/foo`bar.md.txt``" in page_text
    assert "- 知识库：`工程`" in page_text


def test_document_graph_preserves_same_kb_neighbor_under_global_truncation(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    (docs_dir / "a").mkdir(parents=True)
    (docs_dir / "z").mkdir(parents=True)

    cross_links = "\n".join(
        f"- [cross {idx}](../a/doc{idx}.md)" for idx in range(10)
    )
    (docs_dir / "z" / "seed.md").write_text(
        f"""
# Seed

{cross_links}
- [same kb](./zz_neighbor.md)
""".strip(),
        encoding="utf-8",
    )
    (docs_dir / "z" / "zz_neighbor.md").write_text(
        "# Neighbor\n\nSame KB neighbor content.",
        encoding="utf-8",
    )
    for idx in range(10):
        (docs_dir / "a" / f"doc{idx}.md").write_text(
            f"# Cross {idx}\n\nCross KB document {idx}.",
            encoding="utf-8",
        )

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    neighbors = [edge["target"] for edge in bundle.graph_neighbors["z/seed.md"]]
    assert "z/zz_neighbor.md" in neighbors

    expansion = ragbot._expand_candidate_sources_detailed(bundle, ["z/seed.md"], kb="z")
    assert "z/zz_neighbor.md" in expansion.sources


def test_document_graph_keeps_explicit_cross_kb_links_ahead_of_same_series(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    (docs_dir / "foo").mkdir(parents=True)
    (docs_dir / "bar").mkdir(parents=True)

    (docs_dir / "foo" / "seed.md").write_text(
        """
# Seed

See the canonical reference at [linked](../bar/linked.md).
""".strip(),
        encoding="utf-8",
    )
    (docs_dir / "bar" / "linked.md").write_text(
        "# Linked\n\nCross KB explicit target.",
        encoding="utf-8",
    )
    for idx in range(8):
        (docs_dir / "foo" / f"seed_part{idx}.md").write_text(
            f"# Sibling {idx}\n\nSame KB sibling {idx}.",
            encoding="utf-8",
        )

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    neighbors = [edge["target"] for edge in bundle.graph_neighbors["foo/seed.md"]]
    assert "bar/linked.md" in neighbors

    expansion = ragbot._expand_candidate_sources_detailed(bundle, ["foo/seed.md"])
    assert "bar/linked.md" in expansion.sources


def test_document_graph_keeps_links_to_ahead_of_same_kb_mentions_path(tmp_path: Path, monkeypatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    (docs_dir / "foo").mkdir(parents=True)
    (docs_dir / "bar").mkdir(parents=True)

    same_kb_mentions = "\n".join(f"./note{idx}.md" for idx in range(8))
    (docs_dir / "foo" / "seed.md").write_text(
        f"""
# Seed

See the canonical reference at [linked](../bar/linked.md).

{same_kb_mentions}
""".strip(),
        encoding="utf-8",
    )
    (docs_dir / "bar" / "linked.md").write_text(
        "# Linked\n\nCross KB explicit target.",
        encoding="utf-8",
    )
    for idx in range(8):
        (docs_dir / "foo" / f"note{idx}.md").write_text(
            f"# Note {idx}\n\nSame KB path mention target {idx}.",
            encoding="utf-8",
        )

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    neighbors = [edge["target"] for edge in bundle.graph_neighbors["foo/seed.md"]]
    assert "bar/linked.md" in neighbors

    expansion = ragbot._expand_candidate_sources_detailed(bundle, ["foo/seed.md"])
    assert "bar/linked.md" in expansion.sources


def test_document_graph_resolves_relative_links_for_windows_style_sources():
    indexed_files = [
        ragbot.IndexedFile(
            rel_path=r"工程\nested\doc.md",
            suffix=".md",
            kb="工程",
            file_path=Path("doc.md"),
            normalized_text="See [bootstrap](../bootstrap_session.py) for details.",
            chunks=[],
            symbols=[],
        ),
        ragbot.IndexedFile(
            rel_path=r"工程\bootstrap_session.py",
            suffix=".py",
            kb="工程",
            file_path=Path("bootstrap_session.py"),
            normalized_text="def bootstrap_session():\n    pass\n",
            chunks=[],
            symbols=[],
        ),
        ragbot.IndexedFile(
            rel_path=r"产品\bootstrap_session.py",
            suffix=".py",
            kb="产品",
            file_path=Path("bootstrap_session.py"),
            normalized_text="def bootstrap_session():\n    pass\n",
            chunks=[],
            symbols=[],
        ),
    ]

    graph = ragbot._build_document_graph(indexed_files)

    neighbors = [edge["target"] for edge in graph["neighbors"]["工程/nested/doc.md"]]
    assert "工程/bootstrap_session.py" in neighbors
    assert all("\\" not in source for source in graph["neighbors"])


def test_entity_graph_normalizes_windows_style_sources():
    indexed_files = [
        ragbot.IndexedFile(
            rel_path=r"工程\nested\doc.md",
            suffix=".md",
            kb="工程",
            file_path=Path("doc.md"),
            normalized_text="See [bootstrap](../bootstrap_session.py) for details.",
            chunks=[
                ragbot.ChunkSpec(
                    text="See [bootstrap](../bootstrap_session.py) for details.",
                    line_start=1,
                    line_end=1,
                    label="Doc",
                )
            ],
            symbols=[],
        ),
        ragbot.IndexedFile(
            rel_path=r"工程\bootstrap_session.py",
            suffix=".py",
            kb="工程",
            file_path=Path("bootstrap_session.py"),
            normalized_text="def bootstrap_session():\n    pass\n",
            chunks=[
                ragbot.ChunkSpec(
                    text="def bootstrap_session():\n    pass",
                    line_start=1,
                    line_end=2,
                    label="def bootstrap_session",
                )
            ],
            symbols=[
                {
                    "kind": "function",
                    "name": "bootstrap_session",
                    "qualified_name": "bootstrap_session",
                    "source": r"工程\bootstrap_session.py",
                    "line_start": 1,
                    "line_end": 2,
                    "signature": "def bootstrap_session()",
                }
            ],
        ),
    ]

    document_graph = ragbot._build_document_graph(indexed_files)
    entity_graph = ragbot._build_entity_graph(indexed_files, document_graph=document_graph)

    assert all("\\" not in node["source"] for node in entity_graph["nodes"])
    assert any(
        edge["source"] == "section:工程/nested/doc.md:0"
        and edge["target"] == "file:工程/bootstrap_session.py"
        and edge["type"] == "links_to"
        for edge in entity_graph["edges"]
    )


def test_grep_search_prioritizes_exact_config_key(search_bundle):
    result = ragbot.retrieve("api_key 配置项在哪里定义", search_bundle, kb="产品", mode="hybrid")
    assert result["hits"]
    top_hit = result["hits"][0]
    assert top_hit.match_kind == "grep"
    assert top_hit.source == "产品/settings.yaml"
    assert top_hit.line_start is not None


def test_ast_search_finds_python_symbol_with_lines(search_bundle):
    result = ragbot.retrieve("`bootstrap_session()` 函数定义在哪", search_bundle, kb="工程", mode="hybrid")
    assert result["hits"]
    top_hit = result["hits"][0]
    assert top_hit.match_kind == "ast"
    assert top_hit.source == "工程/bootstrap_session.py"
    assert top_hit.line_start == 4
    assert top_hit.line_end >= top_hit.line_start


def test_glob_search_narrows_by_filename(search_bundle):
    result = ragbot.retrieve("看一下 settings.yaml 这个文件", search_bundle, mode="hybrid")
    assert result["hits"]
    assert result["hits"][0].source == "产品/settings.yaml"
    assert any(hit.match_kind == "glob" for hit in result["hits"])


def test_vector_mode_keeps_semantic_recall(search_bundle):
    result = ragbot.retrieve("boot sequence startup orchestration", search_bundle, kb="工程", mode="vector")
    assert result["hits"]
    assert any(hit.source == "工程/startup.md" for hit in result["hits"])
    assert result["search_trace"][0]["stop_reason"] == "vector_mode"


def test_wiki_first_uses_saved_query_note_to_seed_candidate_scope(search_bundle, monkeypatch):
    wiki.save_query_note(
        persist_path=search_bundle.persist_dir,
        question="会话启动链路总览是什么？",
        answer="这是一条会话启动链路总览，串起 playbook 和具体 bootstrap 实现。",
        sources=[
            {
                "source": "工程/session_playbook.md",
                "line_start": 1,
                "line_end": 4,
                "snippet": "Use bootstrap_session() when preparing the concrete initialization steps.",
            },
            {
                "source": "工程/bootstrap_session.py",
                "line_start": 4,
                "line_end": 12,
                "snippet": 'def bootstrap_session(user_id: str) -> dict: """Concrete initialization steps."""',
            },
        ],
    )
    reloaded = ragbot.load_search_bundle(embed_api_key="fake-key", persist_dir=str(search_bundle.persist_dir))
    assert reloaded is not None

    captured: dict[str, set[str]] = {}
    original_vector = ragbot.vector_search

    def tracked_vector(*args, **kwargs):
        captured["candidate_sources"] = set(kwargs.get("candidate_sources") or set())
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(ragbot, "vector_search", tracked_vector)

    result = ragbot.retrieve("会话启动链路总览", reloaded, kb="工程", mode="hybrid")

    assert {"工程/session_playbook.md", "工程/bootstrap_session.py"} <= captured["candidate_sources"]
    assert result["search_trace"][0]["wiki_hits"]
    assert result["search_trace"][0]["wiki_hits"][0]["kind"] == "query"
    assert {"工程/session_playbook.md", "工程/bootstrap_session.py"} <= set(result["search_trace"][0]["wiki_scope"])


def test_retrieve_without_wiki_artifacts_keeps_existing_behavior(search_bundle):
    shutil.rmtree(search_bundle.persist_dir / "wiki")
    reloaded = ragbot.load_search_bundle(embed_api_key="fake-key", persist_dir=str(search_bundle.persist_dir))

    assert reloaded is not None
    assert reloaded.wiki_pages == []

    result = ragbot.retrieve("看一下 settings.yaml 这个文件", reloaded, mode="hybrid")

    assert result["hits"]
    assert result["hits"][0].source == "产品/settings.yaml"
    assert result["search_trace"][0]["wiki_hits"] == []
    assert result["search_trace"][0]["wiki_scope"] == []


def test_agentic_mode_runs_second_step_and_returns_trace(search_bundle, monkeypatch):
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeLLM(*args, **kwargs))
    monkeypatch.setenv("SEARCH_MAX_STEPS", "2")
    monkeypatch.setattr(ragbot, "_should_stop_after_first_step", lambda _hits: (False, "forced_followup"))

    result = ragbot.retrieve(
        "哪里处理用户会话的初始化流程",
        search_bundle,
        kb="工程",
        mode="agentic",
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
    )

    assert len(result["search_trace"]) == 2
    assert result["search_trace"][1]["planner_used"] is True
    assert any(hit.source == "工程/bootstrap_session.py" for hit in result["hits"])


def test_agentic_mode_reuses_step_results_without_duplicate_reruns(search_bundle, monkeypatch):
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeLLM(*args, **kwargs))
    monkeypatch.setenv("SEARCH_MAX_STEPS", "2")
    monkeypatch.setattr(ragbot, "_should_stop_after_first_step", lambda _hits: (False, "forced_followup"))

    counters = {"glob": 0, "grep": 0, "ast": 0, "vector": 0}
    original_glob = ragbot.glob_search
    original_grep = ragbot.grep_search
    original_ast = ragbot.ast_search
    original_vector = ragbot.vector_search

    def counted_glob(*args, **kwargs):
        counters["glob"] += 1
        return original_glob(*args, **kwargs)

    def counted_grep(*args, **kwargs):
        counters["grep"] += 1
        return original_grep(*args, **kwargs)

    def counted_ast(*args, **kwargs):
        counters["ast"] += 1
        return original_ast(*args, **kwargs)

    def counted_vector(*args, **kwargs):
        counters["vector"] += 1
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(ragbot, "glob_search", counted_glob)
    monkeypatch.setattr(ragbot, "grep_search", counted_grep)
    monkeypatch.setattr(ragbot, "ast_search", counted_ast)
    monkeypatch.setattr(ragbot, "vector_search", counted_vector)

    ragbot.retrieve(
        "哪里处理用户会话的初始化流程",
        search_bundle,
        kb="工程",
        mode="agentic",
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
    )

    assert counters == {"glob": 2, "grep": 2, "ast": 2, "vector": 2}


def test_agentic_step2_prefilter_seeds_all_step1_sources(search_bundle, monkeypatch):
    monkeypatch.setenv("SEARCH_MAX_STEPS", "2")
    monkeypatch.setattr(ragbot, "_should_stop_after_first_step", lambda _hits: (False, "forced_followup"))
    monkeypatch.setattr(ragbot, "_call_retrieval_planner", lambda *args, **kwargs: None)

    grouped_hits = {name: [] for name in ragbot.RRF_WEIGHTS}
    step1_hits = [
        ragbot.SearchHit(source="工程/a.md", match_kind="grep", snippet="a-1", score=1.0),
        ragbot.SearchHit(source="工程/a.md", match_kind="vector", snippet="a-2", score=0.9),
        ragbot.SearchHit(source="工程/b.md", match_kind="grep", snippet="b-1", score=0.8),
        ragbot.SearchHit(source="工程/b.md", match_kind="vector", snippet="b-2", score=0.7),
        ragbot.SearchHit(source="工程/c.md", match_kind="grep", snippet="c-1", score=0.6),
        ragbot.SearchHit(source="工程/c.md", match_kind="vector", snippet="c-2", score=0.5),
    ]
    captured: dict[str, object] = {}

    def fake_run_search_step(
        question,
        bundle,
        query_plan,
        kb,
        top_k,
        allowed_sources=None,
        step_name="step1",
        graph_max_hops=1,
        graph_max_extra_sources=12,
    ):
        if step_name == "step1":
            return ragbot.SearchStepResult(
                grouped_hits=grouped_hits,
                hits=step1_hits,
                trace={"step": "step1"},
            )
        captured["allowed_sources"] = allowed_sources
        return ragbot.SearchStepResult(
            grouped_hits=grouped_hits,
            hits=[],
            trace={"step": "step2"},
        )

    def fake_expand(bundle, seed_sources, kb=None, allowed_sources=None, max_hops=1, max_extra_sources=12):
        ordered = list(seed_sources)
        captured["seed_sources"] = ordered
        return ragbot.GraphExpansionResult(
            sources=set(ordered),
            seed_sources=ordered,
            expanded_sources=[],
            edge_reasons=[],
            hops=0,
        )

    monkeypatch.setattr(ragbot, "_run_search_step", fake_run_search_step)
    monkeypatch.setattr(ragbot, "_expand_candidate_sources_detailed", fake_expand)

    ragbot.retrieve(
        "哪里处理初始化流程",
        search_bundle,
        kb="工程",
        mode="agentic",
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
    )

    assert captured["seed_sources"] == ["工程/a.md", "工程/b.md", "工程/c.md"]
    assert captured["allowed_sources"] == {"工程/a.md", "工程/b.md", "工程/c.md"}


def test_run_search_step_preserves_hit_order_for_bounded_graph_expansion(search_bundle, monkeypatch):
    query_plan = ragbot.QueryPlan(
        symbols=[],
        keywords=["session"],
        path_globs=["*session*"],
        semantic_query="session bootstrap flow",
        reason="test ordered seeds",
    )
    glob_hits = [
        ragbot.SearchHit(source="工程/a.md", match_kind="glob", snippet="a", score=0.9),
        ragbot.SearchHit(source="工程/a.md", match_kind="glob", snippet="a duplicate", score=0.8),
        ragbot.SearchHit(source="工程/b.md", match_kind="glob", snippet="b", score=0.7),
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(ragbot, "glob_search", lambda *args, **kwargs: glob_hits)
    monkeypatch.setattr(ragbot, "grep_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(ragbot, "ast_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(ragbot, "_wiki_search", lambda *args, **kwargs: [])

    def fake_expand(bundle, seed_sources, kb=None, allowed_sources=None, max_hops=1, max_extra_sources=12):
        ordered = list(seed_sources)
        captured["seed_sources"] = ordered
        neighbor = "工程/a_neighbor.md" if ordered and ordered[0] == "工程/a.md" else "工程/b_neighbor.md"
        return ragbot.GraphExpansionResult(
            sources=set([*ordered, neighbor]),
            seed_sources=ordered,
            expanded_sources=[neighbor],
            edge_reasons=[{"from": ordered[0], "to": neighbor, "kind": "links_to", "reason": "test", "hop": 1}],
            hops=1,
        )

    def fake_vector_search(bundle, query, kb=None, top_k=8, candidate_sources=None):
        captured["candidate_sources"] = candidate_sources
        return []

    monkeypatch.setattr(ragbot, "_expand_candidate_sources_detailed", fake_expand)
    monkeypatch.setattr(ragbot, "vector_search", fake_vector_search)

    result = ragbot._run_search_step(
        "哪里处理初始化流程",
        search_bundle,
        query_plan,
        kb="工程",
        top_k=4,
        graph_max_hops=1,
        graph_max_extra_sources=1,
    )

    assert captured["seed_sources"] == ["工程/a.md", "工程/b.md"]
    assert "工程/a_neighbor.md" in result.trace["candidate_scope"]
    assert "工程/b_neighbor.md" not in result.trace["candidate_scope"]
    assert result.trace["graph_seed_sources"] == ["工程/a.md", "工程/b.md"]


def test_kb_filter_applies_to_all_retrievers(search_bundle):
    result = ragbot.retrieve("startup", search_bundle, kb="产品", mode="hybrid")
    assert result["hits"]
    assert all(hit.source.startswith("产品/") for hit in result["hits"])


def test_planner_failure_rg_missing_and_ast_failure_fallback(search_bundle, monkeypatch):
    monkeypatch.setenv("SEARCH_MAX_STEPS", "2")
    monkeypatch.setattr(ragbot.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ragbot, "_should_stop_after_first_step", lambda _hits: (False, "forced_followup"))

    class BrokenLLM(FakeLLM):
        def invoke(self, _prompt: str):
            raise RuntimeError("planner failed")

    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: BrokenLLM(*args, **kwargs))

    result = ragbot.retrieve(
        "broken 函数在哪里",
        search_bundle,
        kb="工程",
        mode="agentic",
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
    )

    assert result["hits"]
    assert len(result["search_trace"]) == 2
    assert result["search_trace"][1]["planner_used"] is False
    assert any("invalid.py" in hit.source for hit in result["hits"])


def test_ask_stream_debug_includes_trace(search_bundle, monkeypatch):
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeLLM(*args, **kwargs))

    result = ragbot.ask_stream(
        question="bootstrap_session 在哪",
        search_bundle=search_bundle,
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
        search_mode="agentic",
        debug=True,
    )

    assert "search_trace" in result
    assert "wiki_trace" in result
    assert "".join(result["answer_stream"]) == "mock answer"


def test_ask_stream_debug_includes_bridge_entities(search_bundle, monkeypatch):
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeLLM(*args, **kwargs))
    monkeypatch.setattr(
        ragbot,
        "retrieve",
        lambda *args, **kwargs: {
            "hits": [],
            "context": "ctx",
            "sources": [],
            "search_trace": [
                {
                    "step": "step1",
                    "wiki_hits": [
                        {
                            "kind": "query",
                            "title": "启动链路总览",
                            "relpath": "queries/2026-04-09-startup.md",
                            "score": 3.0,
                            "matched_terms": ["启动"],
                            "source_refs": ["工程/bootstrap_session.py"],
                        }
                    ],
                    "wiki_scope": ["工程/bootstrap_session.py"],
                    "graph_bridge_entities": [
                        {
                            "id": "symbol:工程/bootstrap_session.py:function:bootstrap_session:4",
                            "type": "symbol",
                            "name": "bootstrap_session",
                            "source": "工程/bootstrap_session.py",
                            "relation": "references",
                        }
                    ],
                }
            ],
            "bridge_entities": [
                {
                    "id": "symbol:工程/bootstrap_session.py:function:bootstrap_session:4",
                    "type": "symbol",
                    "name": "bootstrap_session",
                    "source": "工程/bootstrap_session.py",
                    "relation": "references",
                }
            ],
        },
    )

    result = ragbot.ask_stream(
        question="bootstrap_session 在哪",
        search_bundle=search_bundle,
        llm_api_key="fake-key",
        llm_model="fake-model",
        llm_base_url="https://example.com",
        search_mode="agentic",
        debug=True,
    )

    assert "bridge_entities" in result
    assert "wiki_trace" in result
    assert result["bridge_entities"][0]["name"] == "bootstrap_session"
    assert result["wiki_trace"][0]["hit_count"] == 1
    assert result["wiki_trace"][0]["scope"] == ["工程/bootstrap_session.py"]


def test_entity_graph_expansion_prefers_symbol_bridge(tmp_path: Path):
    fake_vectorstore = SimpleNamespace(index=SimpleNamespace(ntotal=0))
    files = [
        {"name": "工程/guide.md", "kb": "工程", "normalized_text": "工程/guide.md.txt"},
        {"name": "工程/bootstrap_session.py", "kb": "工程", "normalized_text": "工程/bootstrap_session.py.txt"},
    ]
    bundle = ragbot.SearchBundle(
        vectorstore=fake_vectorstore,
        persist_dir=tmp_path,
        source_dir=None,
        manifest={"files": files},
        files=files,
        files_by_source={entry["name"]: entry for entry in files},
        normalized_text_dir=tmp_path,
        symbol_index=[],
        document_graph={"version": 1, "edge_count": 0, "neighbors": {}},
        graph_neighbors={},
        entity_graph={"version": 1, "node_count": 4, "edge_count": 2, "nodes": [], "edges": []},
        entity_nodes_by_id={
            "file:工程/guide.md": {
                "id": "file:工程/guide.md",
                "type": "file",
                "name": "guide.md",
                "source": "工程/guide.md",
            },
            "section:工程/guide.md:0": {
                "id": "section:工程/guide.md:0",
                "type": "section",
                "name": "Guide",
                "source": "工程/guide.md",
                "line_start": 1,
                "line_end": 3,
            },
            "symbol:工程/bootstrap_session.py:function:bootstrap_session:4": {
                "id": "symbol:工程/bootstrap_session.py:function:bootstrap_session:4",
                "type": "symbol",
                "name": "bootstrap_session",
                "qualified_name": "bootstrap_session",
                "source": "工程/bootstrap_session.py",
                "line_start": 4,
                "line_end": 7,
            },
            "file:工程/bootstrap_session.py": {
                "id": "file:工程/bootstrap_session.py",
                "type": "file",
                "name": "bootstrap_session.py",
                "source": "工程/bootstrap_session.py",
            },
        },
        entity_edges_by_source={
            "file:工程/guide.md": [
                {
                    "source": "file:工程/guide.md",
                    "target": "section:工程/guide.md:0",
                    "type": "contains",
                    "reason": "Guide",
                }
            ],
            "section:工程/guide.md:0": [
                {
                    "source": "section:工程/guide.md:0",
                    "target": "symbol:工程/bootstrap_session.py:function:bootstrap_session:4",
                    "type": "references",
                    "reason": "bootstrap_session",
                }
            ],
        },
        wiki_pages=[],
    )

    expansion = ragbot._expand_candidate_sources_detailed(bundle, ["工程/guide.md"])

    assert expansion.strategy == "entity_graph"
    assert "工程/bootstrap_session.py" in expansion.sources
    assert expansion.expanded_sources == ["工程/bootstrap_session.py"]
    assert any(entity["name"] == "bootstrap_session" for entity in expansion.bridge_entities)
    assert expansion.edge_reasons[0]["bridges"]


def test_grep_scope_falls_back_to_broader_allowed_sources(search_bundle, monkeypatch):
    monkeypatch.setattr(ragbot, "_wiki_search", lambda *args, **kwargs: [])
    query_plan = ragbot.QueryPlan(
        symbols=[],
        keywords=["api_key"],
        path_globs=["*manual*"],
        semantic_query="api_key",
        reason="test",
    )
    step_result = ragbot._run_search_step(
        question="api_key",
        bundle=search_bundle,
        query_plan=query_plan,
        kb="产品",
        top_k=6,
        allowed_sources={"产品/manual.pdf", "产品/settings.yaml"},
        step_name="fallback-test",
    )

    assert step_result.trace["grep_fallback_used"] is True
    assert step_result.grouped_hits["grep"]
    assert step_result.grouped_hits["grep"][0].source == "产品/settings.yaml"


def test_hybrid_uses_document_graph_to_expand_vector_scope(search_bundle, monkeypatch):
    captured: dict[str, set[str]] = {}
    original_vector = ragbot.vector_search

    def tracked_vector(*args, **kwargs):
        captured["candidate_sources"] = set(kwargs.get("candidate_sources") or set())
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(ragbot, "vector_search", tracked_vector)

    result = ragbot.retrieve("看一下 session_playbook.md 里提到的实现", search_bundle, kb="工程", mode="hybrid")

    assert "工程/bootstrap_session.py" in captured["candidate_sources"]
    trace = result["search_trace"][0]
    assert (
        "工程/bootstrap_session.py" in trace["graph_expanded_sources"]
        or "工程/bootstrap_session.py" in trace["wiki_scope"]
    )


def test_document_graph_respects_kb_boundaries(search_bundle, monkeypatch):
    captured: dict[str, set[str]] = {}
    original_vector = ragbot.vector_search

    def tracked_vector(*args, **kwargs):
        captured["candidate_sources"] = set(kwargs.get("candidate_sources") or set())
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(ragbot, "vector_search", tracked_vector)

    result = ragbot.retrieve("看看 session_bridge.md 提到的背景", search_bundle, kb="产品", mode="hybrid")

    assert all(source.startswith("产品/") for source in captured["candidate_sources"])
    assert "工程/bootstrap_session.py" not in result["search_trace"][0]["graph_expanded_sources"]


def test_extract_query_plan_uses_boundaries_for_extension_aliases():
    broad = ragbot._extract_query_plan("governance 文档怎么写")
    explicit = ragbot._extract_query_plan("看一下 go 文件")

    assert "*.go" not in broad.path_globs
    assert "*.go" in explicit.path_globs


def test_extract_query_plan_promotes_non_ascii_filename_to_exact_keyword_and_glob():
    plan = ragbot._extract_query_plan("创业慧康基础门户系统操作手册.docx")

    assert "创业慧康基础门户系统操作手册.docx" in plan.keywords
    assert "docx" not in {keyword.lower() for keyword in plan.keywords}
    assert plan.symbols == []
    assert "*创业慧康基础门户系统操作手册.docx*" in plan.path_globs
    assert "*.docx" in plan.path_globs


def test_extract_query_plan_promotes_quoted_symbol_tokens():
    plan = ragbot._extract_query_plan("查看 `login` 的定义")
    assert "login" in plan.symbols


def test_extract_query_plan_does_not_treat_qualified_symbol_as_path_glob():
    plan = ragbot._extract_query_plan("查看 `requests.Session.send` 的定义")

    assert "requests.Session.send" in plan.symbols
    assert all("requests.Session.send" not in path_glob for path_glob in plan.path_globs)
    assert ragbot._wiki_source_ref_matches_query("工程/http_client.py", plan) is True


def test_expand_candidate_sources_requires_stronger_token_overlap(search_bundle, monkeypatch):
    monkeypatch.setattr(
        ragbot,
        "_bundle_sources",
        lambda _bundle, kb=None, allowed_sources=None: [
            "工程/bootstrap_session.py",
            "工程/session_helper.py",
            "工程/bootstrap_session_helper.py",
        ],
    )
    monkeypatch.setattr(search_bundle, "graph_neighbors", {})
    monkeypatch.setattr(search_bundle, "entity_graph", {"version": 1, "node_count": 0, "edge_count": 0, "nodes": [], "edges": []})
    monkeypatch.setattr(search_bundle, "entity_nodes_by_id", {})
    monkeypatch.setattr(search_bundle, "entity_edges_by_source", {})

    expanded = ragbot._expand_candidate_sources(search_bundle, {"工程/bootstrap_session.py"})

    assert "工程/bootstrap_session_helper.py" in expanded
    assert "工程/session_helper.py" not in expanded


def test_rg_grep_uses_single_process_for_multiple_keywords(search_bundle, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ragbot.shutil, "which", lambda _name: "/opt/homebrew/bin/rg")

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0,
            stdout="{}:1:def bootstrap_session(user_id: str) -> dict:\n".format(
                search_bundle.cache_path_for("工程/bootstrap_session.py")
            ),
        )

    monkeypatch.setattr(ragbot.subprocess, "run", fake_run)
    hits = ragbot._grep_search_with_rg(
        search_bundle,
        keywords=["bootstrap_session", "user_id", "status"],
        sources=["工程/bootstrap_session.py"],
    )

    assert len(calls) == 1
    assert calls[0].count("-e") == 3
    assert hits


def test_retrieve_prefers_exact_non_ascii_filename_over_other_docx_files(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    (docs_dir / "产品").mkdir(parents=True)
    (docs_dir / "聊天").mkdir(parents=True)

    (docs_dir / "产品" / "创业慧康基础门户系统操作手册.docx").write_text("placeholder", encoding="utf-8")
    (docs_dir / "产品" / "华南大区技术协同规范V1.0.docx").write_text("placeholder", encoding="utf-8")
    (docs_dir / "聊天" / "华南大区群.md").write_text(
        "请重新查看 Hi-HIS-大区技术协同规范V1.0.docx 的流程说明。",
        encoding="utf-8",
    )

    def fake_convert(path: Path) -> str:
        if path.name == "创业慧康基础门户系统操作手册.docx":
            return "# 创业慧康基础门户系统操作手册\n\n系统登录与租户管理。"
        if path.name == "华南大区技术协同规范V1.0.docx":
            return "# 华南大区技术协同规范\n\n大区分支提交流程。"
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "_convert_binary_to_markdown", fake_convert)

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    result = ragbot.retrieve("创业慧康基础门户系统操作手册.docx", bundle, kb="产品", mode="hybrid")
    index_hits = [hit for hit in result["search_trace"][0]["wiki_hits"] if hit["kind"] == "index"]

    assert result["search_trace"][0]["query_plan"]["keywords"] == ["创业慧康基础门户系统操作手册.docx"]
    assert result["search_trace"][0]["wiki_scope"] == ["产品/创业慧康基础门户系统操作手册.docx"]
    assert index_hits and index_hits[0]["source_refs"] == ["产品/创业慧康基础门户系统操作手册.docx"]
    assert result["hits"][0].source == "产品/创业慧康基础门户系统操作手册.docx"
    assert all(hit.source != "聊天/华南大区群.md" for hit in result["hits"][:3])


def test_load_wiki_pages_retries_when_rebuild_temporarily_removes_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    wiki_dir = tmp_path / "wiki" / "queries"
    wiki_dir.mkdir(parents=True)
    page_path = wiki_dir / "2026-04-09-note.md"
    page_path.write_text("# Query Note\n\n[源文件](../files/工程/bootstrap_session.py.md)\n", encoding="utf-8")

    original_read_text = Path.read_text
    call_count = {"page": 0}

    def flaky_read_text(self: Path, *args, **kwargs):
        if self == page_path and call_count["page"] == 0:
            call_count["page"] += 1
            raise FileNotFoundError("transient wiki rebuild window")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)
    monkeypatch.setattr(wiki, "is_wiki_write_in_progress", lambda _persist_path: call_count["page"] == 1)
    monkeypatch.setattr(ragbot.time, "sleep", lambda _seconds: None)

    pages = ragbot._load_wiki_pages(
        tmp_path,
        [{"name": "工程/bootstrap_session.py"}],
    )

    assert pages[0]["kind"] == "query"
    assert pages[0]["relpath"] == "queries/2026-04-09-note.md"
    assert pages[0]["source_refs"] == ["工程/bootstrap_session.py"]
