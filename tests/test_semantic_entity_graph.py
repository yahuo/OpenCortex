from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import ragbot
import ragbot_semantic
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
    def __init__(self, content: str, usage_metadata: dict[str, int] | None = None):
        self.content = content
        self.usage_metadata = usage_metadata or {
            "input_tokens": 20,
            "output_tokens": 8,
            "total_tokens": 28,
        }


class FakeSemanticLLM:
    def __init__(self, call_log: list[str]):
        self.call_log = call_log

    def invoke(self, prompt: str) -> FakeMessage:
        self.call_log.append(prompt)
        lowered = prompt.lower()
        if "settlement handoff" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "settlement handoff",
                                "summary": "Finance close uses a named settlement handoff phase.",
                                "aliases": ["handoff gate"],
                            }
                        ],
                        "decisions": [
                            {
                                "name": "immutable ledger snapshots",
                                "summary": "Finance exports must be based on immutable snapshots.",
                                "aliases": ["ledger snapshotting"],
                                "rationale": ["settlement handoff"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class SlowConcurrentSemanticLLM(FakeSemanticLLM):
    def __init__(
        self,
        call_log: list[str],
        metrics: dict[str, int | threading.Lock],
        delay_seconds: float = 0.05,
    ):
        super().__init__(call_log)
        self.metrics = metrics
        self.delay_seconds = delay_seconds

    def invoke(self, prompt: str) -> FakeMessage:
        lock = self.metrics["lock"]
        assert not isinstance(lock, int)
        with lock:
            self.metrics["active"] = int(self.metrics["active"]) + 1
            self.metrics["max_active"] = max(
                int(self.metrics["max_active"]),
                int(self.metrics["active"]),
            )
        try:
            time.sleep(self.delay_seconds)
            return super().invoke(prompt)
        finally:
            with lock:
                self.metrics["active"] = int(self.metrics["active"]) - 1


class SplitSemanticLLM:
    def __init__(self, call_log: list[str]):
        self.call_log = call_log

    def invoke(self, prompt: str) -> FakeMessage:
        self.call_log.append(prompt)
        lowered = prompt.lower()
        if "concept only marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "shared concept",
                                "summary": "A concept introduced after the decision file.",
                                "aliases": [],
                            }
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "decision only marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [],
                        "decisions": [
                            {
                                "name": "chosen decision",
                                "summary": "A decision that depends on the shared concept.",
                                "aliases": [],
                                "rationale": ["shared concept"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class PunctuationSemanticLLM:
    def __init__(self, call_log: list[str]):
        self.call_log = call_log

    def invoke(self, prompt: str) -> FakeMessage:
        self.call_log.append(prompt)
        lowered = prompt.lower()
        if "punctuation marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {"name": "C#", "summary": "A concept with a sharp suffix.", "aliases": []},
                            {"name": "C++", "summary": "A concept with plus suffixes.", "aliases": []},
                            {"name": "C", "summary": "A plain letter concept.", "aliases": []},
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class AmbiguousAliasSemanticLLM:
    def __init__(self, call_log: list[str]):
        self.call_log = call_log

    def invoke(self, prompt: str) -> FakeMessage:
        self.call_log.append(prompt)
        lowered = prompt.lower()
        if "alias concept a marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "Alpha settlement",
                                "summary": "The alpha concept also uses the settlement alias.",
                                "aliases": ["settlement"],
                            }
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "alias concept b marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [
                            {
                                "name": "Beta settlement",
                                "summary": "The beta concept reuses the same alias.",
                                "aliases": ["settlement"],
                            }
                        ],
                        "decisions": [],
                    },
                    ensure_ascii=False,
                )
            )
        if "alias decision marker" in lowered:
            return FakeMessage(
                json.dumps(
                    {
                        "concepts": [],
                        "decisions": [
                            {
                                "name": "Use settlement policy",
                                "summary": "A decision justified by the shared settlement alias.",
                                "aliases": [],
                                "rationale": ["settlement"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage(json.dumps({"concepts": [], "decisions": []}, ensure_ascii=False))


class QueryNoteSemanticLLM:
    def __init__(self, call_log: list[str]):
        self.call_log = call_log

    def invoke(self, prompt: str) -> FakeMessage:
        self.call_log.append(prompt)
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


def _write_semantic_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "工程").mkdir(parents=True)

    (root / "产品" / "handoff_playbook.md").write_text(
        """
# Handoff Playbook

The settlement handoff is the finance close gate for this workflow.
We keep immutable ledger snapshots so exports can be audited later.
""".strip(),
        encoding="utf-8",
    )
    (root / "工程" / "finalize_payments.py").write_text(
        '''
def finalize_payments(batch_id: str) -> dict:
    """Run the settlement handoff before final export."""
    return {"batch_id": batch_id, "status": "finalized"}
'''.strip(),
        encoding="utf-8",
    )


def _write_split_semantic_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "产品" / "aa_decision_note.md").write_text(
        """
# Decision Note

decision only marker
This file records the chosen decision before any concept page exists.
""".strip(),
        encoding="utf-8",
    )
    (root / "产品" / "zz_concept_note.md").write_text(
        """
# Concept Note

concept only marker
This file introduces the shared concept later in lexical order.
""".strip(),
        encoding="utf-8",
    )


def _write_punctuation_semantic_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "产品" / "language_terms.md").write_text(
        """
# Language Terms

punctuation marker
This note mentions C#, C++, and C as distinct concepts.
""".strip(),
        encoding="utf-8",
    )


def _write_query_note_bridge_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "报告").mkdir(parents=True)
    (root / "产品" / "billing_handoff.md").write_text(
        """
# Billing Handoff

This page explains the billing handoff checklist.
""".strip(),
        encoding="utf-8",
    )
    (root / "报告" / "audit_export.md").write_text(
        """
# Audit Export

This page explains how audit exports are verified.
""".strip(),
        encoding="utf-8",
    )


def _write_query_note_semantic_corpus(root: Path) -> None:
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


def _write_ambiguous_alias_semantic_corpus(root: Path) -> None:
    (root / "产品").mkdir(parents=True)
    (root / "产品" / "alias_concept_a.md").write_text(
        """
# Alias Concept A

alias concept a marker
This file introduces the alpha concept and the shared settlement alias.
""".strip(),
        encoding="utf-8",
    )
    (root / "产品" / "alias_concept_b.md").write_text(
        """
# Alias Concept B

alias concept b marker
This file introduces the beta concept and reuses the same settlement alias.
""".strip(),
        encoding="utf-8",
    )
    (root / "产品" / "alias_decision.md").write_text(
        """
# Alias Decision

alias decision marker
This decision only references the shared settlement alias.
""".strip(),
        encoding="utf-8",
    )


def test_build_vectorstore_writes_semantic_cache_and_stats(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_semantic_corpus(docs_dir)
    llm_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(llm_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
    stats = manifest["semantic_graph_stats"]
    assert stats["enabled"] is True
    assert stats["api_calls"] > 0
    assert stats["total_tokens"] > 0
    assert stats["concurrency"] == 1
    assert (index_dir / "semantic_extract_cache.json").exists()

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in entity_graph["nodes"]}
    edge_types = {edge["type"] for edge in entity_graph["edges"]}
    assert "concept" in node_types
    assert "decision" in node_types
    assert "semantically_related" in edge_types
    assert "rationale_for" in edge_types
    assert llm_calls


def test_semantic_progress_callback_reports_section_progress(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_semantic_corpus(docs_dir)
    llm_calls: list[str] = []
    progress_messages: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(llm_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
        progress_callback=lambda _current, _total, message: progress_messages.append(message),
    )

    semantic_messages = [message for message in progress_messages if message.startswith("语义抽取中 ")]
    assert semantic_messages
    assert semantic_messages[0].startswith("语义抽取中 0/2")
    assert semantic_messages[-1].startswith("语义抽取中 2/2")


def test_semantic_cache_hits_on_rebuild(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_semantic_corpus(docs_dir)
    llm_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(llm_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )
    first_call_count = len(llm_calls)
    assert first_call_count > 0

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )
    second_stats = bundle.manifest["semantic_graph_stats"]
    assert second_stats["cached_sections"] > 0
    assert second_stats["api_calls"] == 0
    assert second_stats["cache_flushes"] == 0
    assert len(llm_calls) == first_call_count


def test_semantic_cache_invalidates_when_base_url_changes(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_semantic_corpus(docs_dir)
    llm_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(llm_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://provider-a.example.com/v1",
    )
    first_call_count = len(llm_calls)
    assert first_call_count > 0

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://provider-b.example.com/v1",
    )
    second_stats = bundle.manifest["semantic_graph_stats"]
    assert second_stats["cached_sections"] == 0
    assert second_stats["api_calls"] > 0
    assert len(llm_calls) > first_call_count


def test_semantic_cache_flushes_incrementally(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_ambiguous_alias_semantic_corpus(docs_dir)
    llm_calls: list[str] = []
    write_calls: list[int] = []

    monkeypatch.setenv("SEMANTIC_CACHE_FLUSH_INTERVAL", "1")
    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(llm_calls),
    )

    original_write_semantic_cache = ragbot_semantic._write_semantic_cache

    def tracked_write_semantic_cache(persist_path: Path, cache_payload: dict[str, object]) -> None:
        entries = cache_payload.get("entries", {})
        assert isinstance(entries, dict)
        write_calls.append(len(entries))
        original_write_semantic_cache(persist_path, cache_payload)

    monkeypatch.setattr(ragbot_semantic, "_write_semantic_cache", tracked_write_semantic_cache)

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    stats = bundle.manifest["semantic_graph_stats"]
    assert write_calls == [1, 2, 3]
    assert stats["cache_flushes"] == len(write_calls)
    assert stats["api_calls"] == 3


def test_semantic_extraction_uses_configured_concurrency(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_semantic_corpus(docs_dir)
    llm_calls: list[str] = []
    metrics: dict[str, int | threading.Lock] = {
        "active": 0,
        "max_active": 0,
        "lock": threading.Lock(),
    }

    monkeypatch.setenv("SEMANTIC_CONCURRENCY", "2")
    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: SlowConcurrentSemanticLLM(llm_calls, metrics),
    )

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    stats = bundle.manifest["semantic_graph_stats"]
    assert stats["concurrency"] == 2
    assert int(metrics["max_active"]) >= 2
    assert stats["api_calls"] == 2


def test_semantic_entity_graph_expansion_adds_cross_file_candidate(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    _write_semantic_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())

    phase2a_dir = tmp_path / "phase2a"
    phase2b_dir = tmp_path / "phase2b"
    semantic_calls: list[str] = []

    phase2a_bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(phase2a_dir),
    )
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: FakeSemanticLLM(semantic_calls),
    )
    phase2b_bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(phase2b_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    seed_source = "产品/handoff_playbook.md"
    target_source = "工程/finalize_payments.py"

    phase2a_expansion = ragbot._expand_candidate_sources_detailed(
        phase2a_bundle,
        [seed_source],
        max_hops=1,
        max_extra_sources=8,
    )
    phase2b_expansion = ragbot._expand_candidate_sources_detailed(
        phase2b_bundle,
        [seed_source],
        max_hops=1,
        max_extra_sources=8,
    )

    assert target_source not in phase2a_expansion.sources
    assert target_source in phase2b_expansion.sources
    assert phase2b_expansion.strategy == "entity_graph"
    assert any(item.get("type") == "concept" for item in phase2b_expansion.bridge_entities)


def test_semantic_rationale_edges_are_order_independent(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_split_semantic_corpus(docs_dir)
    semantic_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: SplitSemanticLLM(semantic_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    node_name_by_id = {
        str(node["id"]): str(node.get("name") or "")
        for node in entity_graph["nodes"]
        if isinstance(node, dict) and node.get("id")
    }
    rationale_edges = {
        (
            node_name_by_id.get(str(edge.get("source", "")), ""),
            node_name_by_id.get(str(edge.get("target", "")), ""),
        )
        for edge in entity_graph["edges"]
        if isinstance(edge, dict) and edge.get("type") == "rationale_for"
    }
    assert ("shared concept", "chosen decision") in rationale_edges
    assert semantic_calls


def test_semantic_entity_graph_expansion_traverses_decision_nodes(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_split_semantic_corpus(docs_dir)
    semantic_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: SplitSemanticLLM(semantic_calls),
    )

    bundle = ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    expansion = ragbot._expand_candidate_sources_detailed(
        bundle,
        ["产品/zz_concept_note.md"],
        max_hops=1,
        max_extra_sources=8,
    )

    assert "产品/aa_decision_note.md" in expansion.sources
    assert any(item.get("type") == "decision" for item in expansion.bridge_entities)


def test_semantic_node_ids_do_not_merge_punctuation_distinct_terms(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_punctuation_semantic_corpus(docs_dir)
    semantic_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: PunctuationSemanticLLM(semantic_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    concept_nodes = [
        node
        for node in entity_graph["nodes"]
        if isinstance(node, dict) and node.get("type") == "concept"
    ]
    concept_names = {str(node.get("name") or "") for node in concept_nodes}
    concept_ids = {str(node.get("id") or "") for node in concept_nodes}
    assert {"C#", "C++", "C"} <= concept_names
    assert len(concept_ids) >= 3
    assert semantic_calls


def test_semantic_rationale_edges_cover_all_matching_alias_concepts(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_ambiguous_alias_semantic_corpus(docs_dir)
    semantic_calls: list[str] = []

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: AmbiguousAliasSemanticLLM(semantic_calls),
    )

    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    node_name_by_id = {
        str(node["id"]): str(node.get("name") or "")
        for node in entity_graph["nodes"]
        if isinstance(node, dict) and node.get("id")
    }
    rationale_edges = {
        (
            node_name_by_id.get(str(edge.get("source", "")), ""),
            node_name_by_id.get(str(edge.get("target", "")), ""),
        )
        for edge in entity_graph["edges"]
        if isinstance(edge, dict) and edge.get("type") == "rationale_for"
    }
    assert ("Alpha settlement", "Use settlement policy") in rationale_edges
    assert ("Beta settlement", "Use settlement policy") in rationale_edges
    assert semantic_calls


def test_save_query_note_refreshes_entity_graph_and_runtime_expansion(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    _write_query_note_bridge_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
    )

    before_bundle = ragbot.load_search_bundle(
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
    )
    assert before_bundle is not None
    before_expansion = ragbot._expand_candidate_sources_detailed(
        before_bundle,
        ["产品/billing_handoff.md"],
        max_hops=1,
        max_extra_sources=8,
    )
    assert "报告/audit_export.md" not in before_expansion.sources

    wiki.save_query_note(
        persist_path=index_dir,
        question="billing handoff 和 audit export 的关系是什么？",
        answer="这两个文件描述的是同一条交付链路的不同环节。",
        sources=[
            {
                "source": "产品/billing_handoff.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This page explains the billing handoff checklist.",
            },
            {
                "source": "报告/audit_export.md",
                "line_start": 1,
                "line_end": 3,
                "snippet": "This page explains how audit exports are verified.",
            },
        ],
    )

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    query_note_nodes = [
        node
        for node in entity_graph["nodes"]
        if isinstance(node, dict) and node.get("type") == "query_note"
    ]
    assert any(node.get("name") == "billing handoff 和 audit export 的关系是什么？" for node in query_note_nodes)

    after_bundle = ragbot.load_search_bundle(
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
    )
    assert after_bundle is not None
    after_expansion = ragbot._expand_candidate_sources_detailed(
        after_bundle,
        ["产品/billing_handoff.md"],
        max_hops=1,
        max_extra_sources=8,
    )
    assert "报告/audit_export.md" in after_expansion.sources
    assert any(item.get("type") == "query_note" for item in after_expansion.bridge_entities)


def test_save_query_note_links_into_semantic_nodes(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "index"
    semantic_calls: list[str] = []
    _write_query_note_semantic_corpus(docs_dir)

    monkeypatch.setattr(ragbot, "make_embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        ragbot,
        "make_llm",
        lambda *args, **kwargs: QueryNoteSemanticLLM(semantic_calls),
    )
    ragbot.build_vectorstore(
        md_dir=str(docs_dir),
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
        llm_api_key="fake-llm-key",
        llm_model="fake-semantic-model",
        llm_base_url="https://example.com/v1",
    )

    before_bundle = ragbot.load_search_bundle(
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
    )
    assert before_bundle is not None
    before_expansion = ragbot._expand_candidate_sources_detailed(
        before_bundle,
        ["运营/faq_bridge.md"],
        max_hops=1,
        max_extra_sources=8,
    )
    assert "工程/concept_consumer.md" not in before_expansion.sources

    wiki.save_query_note(
        persist_path=index_dir,
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

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    node_ids_by_name = {
        str(node.get("name") or ""): str(node.get("id") or "")
        for node in entity_graph["nodes"]
        if isinstance(node, dict) and node.get("id")
    }
    query_note_id = node_ids_by_name["billing chain FAQ 关联哪些实现和决策？"]
    concept_id = node_ids_by_name["billing chain"]
    decision_id = node_ids_by_name["snapshot export policy"]
    semantic_edges = {
        (
            str(edge.get("source", "")),
            str(edge.get("target", "")),
            str(edge.get("type", "")),
        )
        for edge in entity_graph["edges"]
        if isinstance(edge, dict)
    }
    assert (query_note_id, concept_id, "semantically_related") in semantic_edges
    assert (query_note_id, decision_id, "semantically_related") in semantic_edges

    after_bundle = ragbot.load_search_bundle(
        embed_api_key="fake-embed-key",
        persist_dir=str(index_dir),
    )
    assert after_bundle is not None
    after_expansion = ragbot._expand_candidate_sources_detailed(
        after_bundle,
        ["运营/faq_bridge.md"],
        max_hops=1,
        max_extra_sources=8,
    )
    assert "产品/concept_anchor.md" in after_expansion.sources
    assert "报告/decision_playbook.md" in after_expansion.sources
    assert "工程/concept_consumer.md" not in after_expansion.sources
    bridge_types = {str(item.get("type", "")) for item in after_expansion.bridge_entities}
    assert "query_note" in bridge_types
    assert "concept" not in bridge_types
    assert semantic_calls
