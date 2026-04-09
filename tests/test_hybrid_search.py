from __future__ import annotations

import json
from pathlib import Path
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
    assert result["bridge_entities"][0]["name"] == "bootstrap_session"


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
    )

    expansion = ragbot._expand_candidate_sources_detailed(bundle, ["工程/guide.md"])

    assert expansion.strategy == "entity_graph"
    assert "工程/bootstrap_session.py" in expansion.sources
    assert expansion.expanded_sources == ["工程/bootstrap_session.py"]
    assert any(entity["name"] == "bootstrap_session" for entity in expansion.bridge_entities)
    assert expansion.edge_reasons[0]["bridges"]


def test_grep_scope_falls_back_to_broader_allowed_sources(search_bundle):
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
    assert "工程/bootstrap_session.py" in result["search_trace"][0]["graph_expanded_sources"]
    assert result["search_trace"][0]["graph_hops"] == 1


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


def test_extract_query_plan_promotes_quoted_symbol_tokens():
    plan = ragbot._extract_query_plan("查看 `login` 的定义")
    assert "login" in plan.symbols


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
