from __future__ import annotations

import json

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
