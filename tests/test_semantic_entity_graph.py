from __future__ import annotations

import json
from pathlib import Path

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
    assert (index_dir / "semantic_extract_cache.json").exists()

    entity_graph = json.loads((index_dir / "entity_graph.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in entity_graph["nodes"]}
    edge_types = {edge["type"] for edge in entity_graph["edges"]}
    assert "concept" in node_types
    assert "decision" in node_types
    assert "semantically_related" in edge_types
    assert "rationale_for" in edge_types
    assert llm_calls


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
    assert len(llm_calls) == first_call_count


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
