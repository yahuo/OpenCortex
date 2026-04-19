from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ragbot
from ragbot_rerank import _parse_rerank_payload, llm_rerank


def _hit(source: str, snippet: str, score: float = 1.0):
    return ragbot.SearchHit(
        source=source,
        match_kind="vector",
        snippet=snippet,
        score=score,
        line_start=1,
        line_end=2,
    )


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return _FakeMessage(self._payload)


class _ExplodingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("upstream timeout")


def test_parse_rerank_payload_handles_code_fence():
    payload = """```json\n{"rankings": [{"index": 0, "score": 9, "relevant": true}]}\n```"""
    parsed = _parse_rerank_payload(payload, hit_count=1)
    assert parsed == [{"index": 0, "score": 9.0, "relevant": True}]


def test_parse_rerank_payload_filters_out_of_range_indexes():
    payload = json.dumps({"rankings": [{"index": 99, "score": 5}, {"index": 0, "score": 3}]})
    parsed = _parse_rerank_payload(payload, hit_count=2)
    assert parsed == [{"index": 0, "score": 3.0, "relevant": True}]


def test_llm_rerank_disabled_returns_first_keep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "0")
    hits = [_hit(f"a{i}.md", f"snip {i}") for i in range(5)]
    out, trace = llm_rerank(
        "q", hits, llm_api_key="k", llm_model="m", llm_base_url="https://x", keep=3
    )
    assert trace["status"] == "disabled"
    assert [hit.source for hit in out] == ["a0.md", "a1.md", "a2.md"]


def test_llm_rerank_no_credentials_returns_first_keep(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    hits = [_hit(f"a{i}.md", f"snip {i}") for i in range(5)]
    out, trace = llm_rerank(
        "q", hits, llm_api_key="", llm_model="m", llm_base_url="https://x", keep=3
    )
    assert trace["status"] == "no_credentials"
    assert len(out) == 3


def test_llm_rerank_reorders_by_score(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    hits = [
        _hit("low.md", "barely related"),
        _hit("mid.md", "kind of related"),
        _hit("top.md", "very relevant content"),
    ]
    payload = json.dumps(
        {
            "rankings": [
                {"index": 0, "score": 1.0, "relevant": True},
                {"index": 1, "score": 5.0, "relevant": True},
                {"index": 2, "score": 9.0, "relevant": True},
            ]
        }
    )
    fake = _FakeLLM(payload)
    monkeypatch.setattr(ragbot, "make_llm", lambda **_: fake)

    out, trace = llm_rerank(
        "what is X?",
        hits,
        llm_api_key="k",
        llm_model="m",
        llm_base_url="https://x",
        keep=3,
    )
    assert trace["status"] == "ok"
    assert [hit.source for hit in out] == ["top.md", "mid.md", "low.md"]
    assert trace["scores"][0]["source"] == "top.md"
    assert fake.calls, "LLM should have been invoked"


def test_llm_rerank_drops_irrelevant_then_appends_remaining(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    hits = [
        _hit("noise.md", "off topic"),
        _hit("hit.md", "the answer"),
    ]
    payload = json.dumps(
        {
            "rankings": [
                {"index": 0, "score": 0.0, "relevant": False},
                {"index": 1, "score": 8.0, "relevant": True},
            ]
        }
    )
    monkeypatch.setattr(ragbot, "make_llm", lambda **_: _FakeLLM(payload))

    out, trace = llm_rerank(
        "q", hits, llm_api_key="k", llm_model="m", llm_base_url="https://x", keep=2
    )
    assert trace["status"] == "ok"
    # Relevant first, irrelevant tail-appended (we still keep up to `keep`).
    assert out[0].source == "hit.md"
    assert any(hit.source == "noise.md" for hit in out)


def test_llm_rerank_falls_back_on_exception(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    hits = [_hit(f"a{i}.md", f"snip {i}") for i in range(4)]
    monkeypatch.setattr(ragbot, "make_llm", lambda **_: _ExplodingLLM())

    out, trace = llm_rerank(
        "q", hits, llm_api_key="k", llm_model="m", llm_base_url="https://x", keep=2
    )
    assert trace["status"] == "fallback"
    assert "RuntimeError" in trace.get("error", "")
    assert [hit.source for hit in out] == ["a0.md", "a1.md"]


def test_llm_rerank_falls_back_on_unparseable_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    hits = [_hit(f"a{i}.md", f"snip {i}") for i in range(3)]
    monkeypatch.setattr(ragbot, "make_llm", lambda **_: _FakeLLM("not json at all"))

    out, trace = llm_rerank(
        "q", hits, llm_api_key="k", llm_model="m", llm_base_url="https://x", keep=2
    )
    assert trace["status"] == "fallback"
    assert [hit.source for hit in out] == ["a0.md", "a1.md"]


def test_llm_rerank_uses_rerank_model_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_ENABLED", "1")
    monkeypatch.setenv("RERANK_MODEL", "cheap-model")
    hits = [_hit("a.md", "x")]
    payload = json.dumps({"rankings": [{"index": 0, "score": 5.0, "relevant": True}]})
    captured: dict[str, str] = {}

    def fake_make_llm(**kwargs):
        captured.update(kwargs)
        return _FakeLLM(payload)

    monkeypatch.setattr(ragbot, "make_llm", fake_make_llm)
    _, trace = llm_rerank(
        "q", hits, llm_api_key="k", llm_model="default-model", llm_base_url="https://x"
    )
    assert captured["model"] == "cheap-model"
    assert trace["model"] == "cheap-model"


def test_llm_rerank_no_hits_returns_empty():
    out, trace = llm_rerank(
        "q", [], llm_api_key="k", llm_model="m", llm_base_url="https://x"
    )
    assert out == []
    assert trace["status"] == "no_hits"


def test_retrieve_missing_rerank_credentials_matches_disabled_behavior(monkeypatch: pytest.MonkeyPatch):
    bundle = SimpleNamespace()

    def fake_vector_search(_bundle, _query, kb=None, top_k=6, candidate_sources=None):
        return [_hit(f"doc-topk-{top_k}-{i}.md", f"snip {top_k}-{i}") for i in range(top_k)]

    monkeypatch.setattr(ragbot, "vector_search", fake_vector_search)
    monkeypatch.setenv("RERANK_TOP_N", "30")

    monkeypatch.setenv("RERANK_ENABLED", "0")
    disabled = ragbot.retrieve(
        "q", bundle, mode="vector", top_k=3, llm_api_key="", llm_model="m", llm_base_url=""
    )

    monkeypatch.setenv("RERANK_ENABLED", "1")
    no_credentials = ragbot.retrieve(
        "q", bundle, mode="vector", top_k=3, llm_api_key="", llm_model="m", llm_base_url=""
    )

    assert [hit.source for hit in disabled["hits"]] == [hit.source for hit in no_credentials["hits"]]
    assert disabled["search_trace"][0]["rerank"]["status"] == "disabled"
    assert no_credentials["search_trace"][0]["rerank"]["status"] == "no_credentials"


def test_retrieve_refreshes_trace_top_sources_after_rerank(monkeypatch: pytest.MonkeyPatch):
    bundle = SimpleNamespace()
    initial_hits = [
        _hit("a.md", "first"),
        _hit("b.md", "second"),
        _hit("c.md", "third"),
    ]

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
        return ragbot.SearchStepResult(
            grouped_hits={name: [] for name in ragbot.RRF_WEIGHTS},
            hits=list(initial_hits),
            trace={
                "step": step_name,
                "top_sources": [hit.source for hit in initial_hits[:3]],
                "graph_bridge_entities": [],
            },
        )

    def fake_rerank(question, hits, **kwargs):
        return [hits[2], hits[1], hits[0]], {"status": "ok", "kept": 3, "scores": []}

    monkeypatch.setattr(ragbot, "_run_search_step", fake_run_search_step)
    monkeypatch.setattr(ragbot, "llm_rerank", fake_rerank)
    monkeypatch.setenv("RERANK_ENABLED", "1")

    result = ragbot.retrieve(
        "q", bundle, mode="hybrid", top_k=3, llm_api_key="k", llm_model="m", llm_base_url="https://x"
    )

    assert [hit.source for hit in result["hits"]] == ["c.md", "b.md", "a.md"]
    assert result["search_trace"][-1]["top_sources"] == ["c.md", "b.md", "a.md"]
    assert result["search_trace"][-1]["rerank"]["status"] == "ok"


def test_retrieve_uses_env_top_k_when_omitted(monkeypatch: pytest.MonkeyPatch):
    bundle = SimpleNamespace()

    def fake_vector_search(_bundle, _query, kb=None, top_k=6, candidate_sources=None):
        return [_hit(f"doc-topk-{top_k}-{i}.md", f"snip {top_k}-{i}") for i in range(top_k)]

    monkeypatch.setattr(ragbot, "vector_search", fake_vector_search)
    monkeypatch.setenv("SEARCH_TOP_K", "4")
    monkeypatch.setenv("RERANK_ENABLED", "0")

    result = ragbot.retrieve(
        "q", bundle, mode="vector", llm_api_key="", llm_model="m", llm_base_url=""
    )

    assert [hit.source for hit in result["hits"]] == [
        "doc-topk-4-0.md",
        "doc-topk-4-1.md",
        "doc-topk-4-2.md",
        "doc-topk-4-3.md",
    ]
