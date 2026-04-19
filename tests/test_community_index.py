from __future__ import annotations

import json
from pathlib import Path

import pytest

import ragbot
import wiki


@pytest.fixture(autouse=True)
def _enable_semantic_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_GRAPH_ENABLED", "1")


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
        self.usage_metadata = {"input_tokens": 18, "output_tokens": 7, "total_tokens": 25}


class FakeSemanticLLM:
    def invoke(self, prompt: str):
        lowered = prompt.lower()
        if "orchestration" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "orchestration phase",
                                "summary": "A shared orchestration concept across docs.",
                                "aliases": [],
                            }
                        ],
                        "decisions": [
                            {
                                "name": "deferred report archival",
                                "summary": "Reporting work is deferred until orchestration completes.",
                                "aliases": [],
                                "rationale": ["orchestration phase"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class SplitCommunitySemanticLLM:
    def invoke(self, prompt: str):
        lowered = prompt.lower()
        if "concept bridge marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "shared semantic bridge",
                                "summary": "A concept documented in one file and referenced elsewhere.",
                                "aliases": [],
                            }
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "decision bridge marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [],
                        "decisions": [
                            {
                                "name": "bridge rollout policy",
                                "summary": "A decision that depends on the shared bridge concept.",
                                "aliases": [],
                                "rationale": ["shared semantic bridge"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class QueryNoteCommunitySemanticLLM:
    def invoke(self, prompt: str):
        lowered = prompt.lower()
        if "note concept marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "billing chain",
                                "summary": "A shared concept reused across billing docs.",
                                "aliases": ["handoff chain"],
                            }
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "note decision marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [],
                        "decisions": [
                            {
                                "name": "snapshot export policy",
                                "summary": "Exports are reviewed from immutable snapshots.",
                                "aliases": [],
                                "rationale": ["billing chain"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


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


def _write_split_community_semantic_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "报告").mkdir(parents=True)

    (root / "产品" / "concept_bridge.md").write_text(
        """
# Concept Bridge

concept bridge marker
This document introduces the shared semantic bridge concept.
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "decision_bridge.md").write_text(
        """
# Decision Bridge

    decision bridge marker
This report records the rollout policy but does not repeat the concept page path.
""".strip(),
        encoding="utf-8",
    )


def _write_query_note_community_corpus(root: Path) -> None:
    (root / "工程").mkdir(parents=True)
    (root / "报告").mkdir(parents=True)

    (root / "工程" / "handoff_checklist.md").write_text(
        """
# Handoff Checklist

This document describes the billing handoff checklist.
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "export_review.md").write_text(
        """
# Export Review

This document describes the audit export review workflow.
""".strip(),
        encoding="utf-8",
    )


def _write_query_note_semantic_community_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "工程").mkdir(parents=True)
    (root / "报告").mkdir(parents=True)
    (root / "运营").mkdir(parents=True)
    (root / "产品" / "concept_anchor.md").write_text(
        """
# Concept Anchor

note concept marker
This file introduces the shared billing chain concept.
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "concept_consumer.md").write_text(
        """
# Concept Consumer

note concept marker
This file reuses the shared billing chain concept in engineering.
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "decision_playbook.md").write_text(
        """
# Decision Playbook

note decision marker
This file records the snapshot export policy used in reporting.
""".strip(),
        encoding="utf-8",
    )
    (root / "运营" / "faq_bridge.md").write_text(
        """
# FAQ Bridge

This page captures the saved question but contains no semantic marker by itself.
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

    graph_report_path = bundle.persist_dir / "reports" / "GRAPH_REPORT.md"
    graph_report = graph_report_path.read_text(encoding="utf-8")
    assert "## God Nodes" in graph_report
    assert "### community-001: bootstrap_session" in graph_report
    assert "bootstrap_session @ 工程/bootstrap_session.py" in graph_report
    assert "archive_report @ 报告/archive_report.py" in graph_report
    assert "工程/session_notes.md -> 报告/report_notes.md" in graph_report
    assert "bootstrap_session 在哪里实现？" in graph_report


def test_build_vectorstore_surfaces_semantic_summary_in_community_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeSemanticLLM())

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    community_index = json.loads((bundle.persist_dir / "community_index.json").read_text(encoding="utf-8"))
    semantic_summary = community_index["semantic_summary"]
    assert semantic_summary["enabled"] is True
    assert semantic_summary["concept_count"] >= 1
    assert semantic_summary["decision_count"] >= 1
    assert semantic_summary["api_calls"] > 0

    assert any(
        any(item["name"] == "orchestration phase" for item in community.get("top_concepts", []))
        for community in community_index["communities"]
    )
    assert any(
        any(item["name"] == "deferred report archival" for item in community.get("top_decisions", []))
        for community in community_index["communities"]
    )

    graph_report = (bundle.persist_dir / "reports" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "## 语义抽取" in graph_report
    assert "orchestration phase" in graph_report
    assert "deferred report archival" in graph_report


def test_build_vectorstore_writes_wiki_community_and_entity_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeSemanticLLM())

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    community_page = bundle.persist_dir / "wiki" / "communities" / "community-001.md"
    assert community_page.exists()
    community_text = community_page.read_text(encoding="utf-8")
    assert "Top Files" in community_text
    assert "Top Concepts" in community_text
    assert "orchestration phase" in community_text

    entity_pages = sorted((bundle.persist_dir / "wiki" / "entities").glob("*.md"))
    assert entity_pages
    assert any("orchestration phase" in page.read_text(encoding="utf-8") for page in entity_pages)

    wiki_index = (bundle.persist_dir / "wiki" / "index.md").read_text(encoding="utf-8")
    assert "## Communities" in wiki_index
    assert "## Entities" in wiki_index

    reloaded = ragbot.load_search_bundle(embed_api_key="fake-key", persist_dir=str(index_dir))
    assert reloaded is not None
    assert any(page["kind"] == "community" for page in reloaded.wiki_pages)
    assert any(page["kind"] == "entity" for page in reloaded.wiki_pages)


def test_wiki_first_runtime_uses_entity_pages_to_seed_semantic_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: FakeSemanticLLM())

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )
    reloaded = ragbot.load_search_bundle(embed_api_key="fake-key", persist_dir=str(index_dir))
    assert reloaded is not None

    captured: dict[str, set[str]] = {}
    original_vector = ragbot.vector_search

    def tracked_vector(*args, **kwargs):
        captured["candidate_sources"] = set(kwargs.get("candidate_sources") or set())
        return original_vector(*args, **kwargs)

    monkeypatch.setattr(ragbot, "vector_search", tracked_vector)

    result = ragbot.retrieve("orchestration phase 在哪些文件里被讨论？", reloaded, mode="hybrid")

    assert {"工程/session_notes.md", "报告/report_notes.md"} <= captured["candidate_sources"]
    assert any(
        hit["kind"] == "entity" and hit["title"] == "orchestration phase"
        for hit in result["search_trace"][0]["wiki_hits"]
    )


def test_build_vectorstore_projects_rationale_only_semantic_bridges_into_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_split_community_semantic_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: SplitCommunitySemanticLLM())

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    community_index = json.loads((bundle.persist_dir / "community_index.json").read_text(encoding="utf-8"))
    assert community_index["community_count"] == 2
    assert any(
        bridge["kind"] == "rationale_for"
        and bridge["source"] == "产品/concept_bridge.md"
        and bridge["target"] == "报告/decision_bridge.md"
        for bridge in community_index["bridges"]
    )

    graph_report = (bundle.persist_dir / "reports" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "rationale_for | 产品/concept_bridge.md -> 报告/decision_bridge.md" in graph_report


def test_save_query_note_refreshes_community_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_query_note_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
    )

    wiki.save_query_note(
        persist_path=bundle.persist_dir,
        question="handoff checklist 和 export review 是同一条链路吗？",
        answer="是，它们分别覆盖交付前检查和交付后复核。",
        sources=[
            {
                "source": "工程/handoff_checklist.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This document describes the billing handoff checklist.",
            },
            {
                "source": "报告/export_review.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This document describes the audit export review workflow.",
            },
        ],
    )

    community_index = json.loads((bundle.persist_dir / "community_index.json").read_text(encoding="utf-8"))
    assert community_index["semantic_summary"]["query_note_count"] == 1
    assert any(
        bridge["kind"] == "semantically_related"
        and bridge["source"] == "工程/handoff_checklist.md"
        and bridge["target"] == "报告/export_review.md"
        and any(item.get("type") == "query_note" for item in bridge.get("bridges", []))
        for bridge in community_index["bridges"]
    )
    assert any(
        any(item["name"] == "handoff checklist 和 export review 是同一条链路吗？" for item in community.get("top_query_notes", []))
        for community in community_index["communities"]
    )

    reloaded = ragbot.load_search_bundle(embed_api_key="fake-key", persist_dir=str(bundle.persist_dir))
    assert reloaded is not None
    assert any(
        page.get("kind") == "community"
        and "handoff checklist 和 export review 是同一条链路吗？" in str(page.get("text", ""))
        for page in reloaded.wiki_pages
    )

    graph_report = (bundle.persist_dir / "reports" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "Top Query Notes" in graph_report
    assert "handoff checklist 和 export review 是同一条链路吗？" in graph_report


def test_query_note_semantic_links_project_into_community_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_query_note_semantic_community_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(ragbot, "make_llm", lambda *args, **kwargs: QueryNoteCommunitySemanticLLM())
    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    wiki.save_query_note(
        persist_path=bundle.persist_dir,
        question="billing chain FAQ 关联哪些实现和决策？",
        answer="它关联了共享概念和 export snapshot 决策。",
        sources=[
            {
                "source": "运营/faq_bridge.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This page captures the saved question but contains no semantic marker by itself.",
            },
            {
                "source": "产品/concept_anchor.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This file introduces the shared billing chain concept.",
            },
            {
                "source": "报告/decision_playbook.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This file records the snapshot export policy used in reporting.",
            },
        ],
    )

    community_index = json.loads((bundle.persist_dir / "community_index.json").read_text(encoding="utf-8"))
    assert not any(
        bridge["kind"] == "semantically_related"
        and {bridge["source"], bridge["target"]} == {"运营/faq_bridge.md", "工程/concept_consumer.md"}
        for bridge in community_index["bridges"]
    )
    assert any(
        bridge["kind"] == "semantically_related"
        and {bridge["source"], bridge["target"]} == {"产品/concept_anchor.md", "工程/concept_consumer.md"}
        and any(item.get("type") == "query_note" for item in bridge.get("bridges", []))
        and any(item.get("type") == "concept" for item in bridge.get("bridges", []))
        for bridge in community_index["bridges"]
    )

    graph_report = (bundle.persist_dir / "reports" / "GRAPH_REPORT.md").read_text(encoding="utf-8")
    assert "billing chain FAQ 关联哪些实现和决策？" in graph_report
    assert (
        "产品/concept_anchor.md -> 工程/concept_consumer.md" in graph_report
        or "工程/concept_consumer.md -> 产品/concept_anchor.md" in graph_report
    )
