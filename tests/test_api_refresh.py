from __future__ import annotations

import api


def test_ask_refreshes_bundle_when_artifacts_change(monkeypatch) -> None:
    first_bundle = object()
    second_bundle = object()
    refreshed_signature = (("entity_graph.json", True, 2, 128),)
    captured: dict[str, object] = {}

    api._cfg = {
        "persist_dir": "/tmp/fake-index",
        "embed_api_key": "fake-embed-key",
        "embed_base_url": "https://example.com/embed",
        "embed_model": "fake-embed-model",
        "llm_api_key": "fake-llm-key",
        "llm_base_url": "https://example.com/llm",
        "llm_model": "fake-llm-model",
        "search_mode": "hybrid",
    }
    api._search_bundle = first_bundle
    api._search_bundle_signature = (("entity_graph.json", True, 1, 64),)

    monkeypatch.setattr(api, "_search_artifact_signature", lambda _persist_dir: refreshed_signature)
    monkeypatch.setattr(api, "load_search_bundle", lambda **_kwargs: second_bundle)

    def fake_ask_stream(*, search_bundle, **_kwargs):
        captured["bundle"] = search_bundle
        return {"answer_stream": iter(["ok"]), "sources": []}

    monkeypatch.setattr(api, "rag_ask_stream", fake_ask_stream)

    response = api.ask(api.AskRequest(question="保存后的 query note 会生效吗？", stream=False))

    assert response == {"answer": "ok", "sources": []}
    assert captured["bundle"] is second_bundle
    assert api._search_bundle is second_bundle
    assert api._search_bundle_signature == refreshed_signature


def test_refresh_retries_until_artifact_signature_stabilizes(monkeypatch) -> None:
    old_bundle = object()
    unstable_bundle = object()
    stable_bundle = object()
    old_signature = (("entity_graph.json", True, 1, 64),)
    partial_signature = (("entity_graph.json", True, 2, 32),)
    complete_signature = (("entity_graph.json", True, 3, 128),)

    api._cfg = {
        "persist_dir": "/tmp/fake-index",
        "embed_api_key": "fake-embed-key",
        "embed_base_url": "https://example.com/embed",
        "embed_model": "fake-embed-model",
        "llm_api_key": "fake-llm-key",
        "llm_base_url": "https://example.com/llm",
        "llm_model": "fake-llm-model",
        "search_mode": "hybrid",
    }
    api._search_bundle = old_bundle
    api._search_bundle_signature = old_signature

    signature_sequence = iter(
        [
            partial_signature,
            partial_signature,
            complete_signature,
            complete_signature,
            complete_signature,
        ]
    )
    monkeypatch.setattr(api, "_search_artifact_signature", lambda _persist_dir: next(signature_sequence))
    load_calls: list[object] = []

    def fake_load_search_bundle(**_kwargs):
        bundle = unstable_bundle if not load_calls else stable_bundle
        load_calls.append(bundle)
        return bundle

    monkeypatch.setattr(api, "load_search_bundle", fake_load_search_bundle)
    monkeypatch.setattr(api.time, "sleep", lambda _seconds: None)

    refreshed = api._refresh_search_bundle_if_needed()

    assert refreshed is stable_bundle
    assert load_calls == [unstable_bundle, stable_bundle]
    assert api._search_bundle is stable_bundle
    assert api._search_bundle_signature == complete_signature


def test_refresh_falls_back_to_previous_bundle_when_reload_raises(monkeypatch) -> None:
    old_bundle = object()
    old_signature = (("entity_graph.json", True, 1, 64),)
    changed_signature = (("entity_graph.json", True, 2, 128),)

    api._cfg = {
        "persist_dir": "/tmp/fake-index",
        "embed_api_key": "fake-embed-key",
        "embed_base_url": "https://example.com/embed",
        "embed_model": "fake-embed-model",
        "llm_api_key": "fake-llm-key",
        "llm_base_url": "https://example.com/llm",
        "llm_model": "fake-llm-model",
        "search_mode": "hybrid",
    }
    api._search_bundle = old_bundle
    api._search_bundle_signature = old_signature

    monkeypatch.setattr(api, "_search_artifact_signature", lambda _persist_dir: changed_signature)
    monkeypatch.setattr(api.time, "sleep", lambda _seconds: None)

    def fake_load_search_bundle(**_kwargs):
        raise ValueError("manifest is temporarily invalid")

    monkeypatch.setattr(api, "load_search_bundle", fake_load_search_bundle)

    refreshed = api._refresh_search_bundle_if_needed()

    assert refreshed is old_bundle
    assert api._search_bundle is old_bundle
    assert api._search_bundle_signature == old_signature
