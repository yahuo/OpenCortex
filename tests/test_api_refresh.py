from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import api


def _fake_bundle(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        persist_dir=tmp_path,
        manifest={"graph_report_file": "reports/GRAPH_REPORT.md"},
        wiki_pages=[
            {
                "id": "wiki:index:index.md",
                "kind": "index",
                "title": "知识导航",
                "relpath": "index.md",
                "text": "# 知识导航\n",
                "source_refs": ["工程/bootstrap_session.py"],
            },
            {
                "id": "wiki:community:communities/session-bootstrap.md",
                "kind": "community",
                "title": "Session Bootstrap",
                "relpath": "communities/session-bootstrap.md",
                "text": "# Session Bootstrap\n",
                "source_refs": ["工程/bootstrap_session.py", "工程/session_playbook.md"],
            },
            {
                "id": "wiki:query:queries/2026-04-09-session-bootstrap.md",
                "kind": "query",
                "title": "session bootstrap",
                "relpath": "queries/2026-04-09-session-bootstrap.md",
                "text": "# session bootstrap\n",
                "source_refs": ["工程/bootstrap_session.py"],
            },
        ],
    )


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


def test_api_debug_response_exposes_wiki_trace(monkeypatch) -> None:
    bundle = _fake_bundle(Path("/tmp/fake-index"))
    signature = (("entity_graph.json", True, 1, 64),)

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
    api._search_bundle = bundle
    api._search_bundle_signature = signature

    monkeypatch.setattr(api, "_search_artifact_signature", lambda _persist_dir: signature)
    monkeypatch.setattr(
        api,
        "rag_ask_stream",
        lambda **_kwargs: {
            "answer_stream": iter(["ok"]),
            "sources": [],
            "bridge_entities": [],
            "wiki_trace": [{"step": "step1", "hit_count": 1, "scope_count": 1, "hits": [], "scope": ["工程/bootstrap_session.py"]}],
            "search_trace": [{"step": "step1"}],
        },
    )

    response = api.ask(api.AskRequest(question="query note 生效了吗？", stream=False, debug=True))

    assert response["answer"] == "ok"
    assert response["wiki_trace"][0]["scope"] == ["工程/bootstrap_session.py"]
    assert response["artifacts"]["wiki_pages"][0]["relpath"] == "index.md"
    assert response["artifacts"]["community_pages"][0]["relpath"] == "communities/session-bootstrap.md"


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


def test_refresh_waits_until_wiki_write_lock_releases(monkeypatch) -> None:
    old_bundle = object()
    refreshed_bundle = object()
    old_signature = (("entity_graph.json", True, 1, 64),)
    refreshed_signature = (("entity_graph.json", True, 2, 128),)

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

    lock_states = iter([True, False, False])
    load_calls: list[object] = []

    monkeypatch.setattr(api, "_search_artifact_signature", lambda _persist_dir: refreshed_signature)
    monkeypatch.setattr(api, "_wiki_write_in_progress", lambda _persist_dir: next(lock_states))
    monkeypatch.setattr(api.time, "sleep", lambda _seconds: None)

    def fake_load_search_bundle(**_kwargs):
        load_calls.append(refreshed_bundle)
        return refreshed_bundle

    monkeypatch.setattr(api, "load_search_bundle", fake_load_search_bundle)

    refreshed = api._refresh_search_bundle_if_needed()

    assert refreshed is refreshed_bundle
    assert load_calls == [refreshed_bundle]
    assert api._search_bundle is refreshed_bundle
    assert api._search_bundle_signature == refreshed_signature


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


def test_wiki_index_returns_pages_and_counts(monkeypatch, tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(api, "_refresh_search_bundle_if_needed", lambda force=False: bundle)

    response = api.wiki_index()

    assert response["index"]["relpath"] == "index.md"
    assert response["counts"] == {"index": 1, "community": 1, "query": 1}
    assert len(response["pages"]) == 3


def test_wiki_page_returns_matching_content(monkeypatch, tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(api, "_refresh_search_bundle_if_needed", lambda force=False: bundle)

    response = api.wiki_page("communities/session-bootstrap.md")

    assert response["page"]["kind"] == "community"
    assert "Session Bootstrap" in response["page"]["content"]


def test_wiki_page_rejects_path_traversal(monkeypatch, tmp_path: Path) -> None:
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(api, "_refresh_search_bundle_if_needed", lambda force=False: bundle)

    with pytest.raises(api.HTTPException) as exc_info:
        api.wiki_page("../secrets.md")

    assert exc_info.value.status_code == 400


def test_save_query_endpoint_refreshes_bundle_and_returns_note_page(monkeypatch, tmp_path: Path) -> None:
    initial_bundle = _fake_bundle(tmp_path)
    refreshed_bundle = SimpleNamespace(
        persist_dir=tmp_path,
        manifest={"graph_report_file": "reports/GRAPH_REPORT.md"},
        wiki_pages=initial_bundle.wiki_pages
        + [
            {
                "id": "wiki:query:queries/2026-04-09-new-note.md",
                "kind": "query",
                "title": "新的知识笔记",
                "relpath": "queries/2026-04-09-new-note.md",
                "text": "# 新的知识笔记\n",
                "source_refs": ["工程/bootstrap_session.py"],
            }
        ],
    )
    refresh_calls: list[bool] = []
    saved: dict[str, object] = {}

    api._cfg = {"persist_dir": str(tmp_path)}

    def fake_save_query_note(**kwargs):
        saved.update(kwargs)
        return {
            "note_path": str(tmp_path / "wiki" / "queries" / "2026-04-09-new-note.md"),
            "note_relpath": "wiki/queries/2026-04-09-new-note.md",
            "log_path": str(tmp_path / "wiki" / "log.md"),
        }

    def fake_refresh(force: bool = False):
        refresh_calls.append(force)
        return refreshed_bundle

    monkeypatch.setattr(api, "save_query_note", fake_save_query_note)
    monkeypatch.setattr(api, "_refresh_search_bundle_if_needed", fake_refresh)

    response = api.save_query(
        api.SaveQueryRequest(
            question="保存 query note",
            answer="已完成",
            sources=[{"source": "工程/bootstrap_session.py"}],
            tags=["验收"],
        )
    )

    assert saved["persist_path"] == tmp_path
    assert refresh_calls == [False]
    assert response["note"]["note_relpath"] == "wiki/queries/2026-04-09-new-note.md"
    assert response["page"]["relpath"] == "queries/2026-04-09-new-note.md"


def test_graph_report_returns_markdown_content(monkeypatch, tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True)
    (report_dir / "GRAPH_REPORT.md").write_text("# 图谱报告\n\n内容", encoding="utf-8")
    bundle = _fake_bundle(tmp_path)
    monkeypatch.setattr(api, "_refresh_search_bundle_if_needed", lambda force=False: bundle)

    response = api.graph_report()

    assert response["path"] == "reports/GRAPH_REPORT.md"
    assert response["content"].startswith("# 图谱报告")


def test_search_artifact_signature_tracks_runtime_wiki_pages(tmp_path: Path) -> None:
    (tmp_path / "index_manifest.json").write_text("{}", encoding="utf-8")
    wiki_dir = tmp_path / "wiki"
    (wiki_dir / "queries").mkdir(parents=True)
    (wiki_dir / "communities").mkdir(parents=True)
    (wiki_dir / "index.md").write_text("# 知识导航\n", encoding="utf-8")
    (wiki_dir / "queries" / "2026-04-09-note.md").write_text("# Query\n", encoding="utf-8")
    (wiki_dir / "communities" / "community-1.md").write_text("# Community\n", encoding="utf-8")
    (wiki_dir / "log.md").write_text("# Log\n", encoding="utf-8")

    signature = api._search_artifact_signature(str(tmp_path))
    signature_paths = {item[0] for item in signature}

    assert "wiki/index.md" in signature_paths
    assert "wiki/queries/2026-04-09-note.md" in signature_paths
    assert "wiki/communities/community-1.md" in signature_paths
    assert "wiki/log.md" not in signature_paths


def test_refresh_reloads_bundle_when_only_runtime_wiki_page_changes(monkeypatch, tmp_path: Path) -> None:
    persist_dir = tmp_path / "index"
    persist_dir.mkdir(parents=True)
    (persist_dir / "index_manifest.json").write_text("{}", encoding="utf-8")
    wiki_dir = persist_dir / "wiki" / "queries"
    wiki_dir.mkdir(parents=True)
    page_path = wiki_dir / "2026-04-09-note.md"
    page_path.write_text("# before\n", encoding="utf-8")

    first_bundle = object()
    second_bundle = object()
    captured: dict[str, object] = {}

    api._cfg = {
        "persist_dir": str(persist_dir),
        "embed_api_key": "fake-embed-key",
        "embed_base_url": "https://example.com/embed",
        "embed_model": "fake-embed-model",
        "llm_api_key": "fake-llm-key",
        "llm_base_url": "https://example.com/llm",
        "llm_model": "fake-llm-model",
        "search_mode": "hybrid",
    }
    api._search_bundle = first_bundle
    api._search_bundle_signature = api._search_artifact_signature(str(persist_dir))

    page_path.write_text("# after\n", encoding="utf-8")
    changed_signature = api._search_artifact_signature(str(persist_dir))
    assert changed_signature != api._search_bundle_signature

    monkeypatch.setattr(api, "load_search_bundle", lambda **_kwargs: second_bundle)

    def fake_ask_stream(*, search_bundle, **_kwargs):
        captured["bundle"] = search_bundle
        return {"answer_stream": iter(["ok"]), "sources": []}

    monkeypatch.setattr(api, "rag_ask_stream", fake_ask_stream)

    response = api.ask(api.AskRequest(question="wiki page 修改后会刷新吗？", stream=False))

    assert response == {"answer": "ok", "sources": []}
    assert captured["bundle"] is second_bundle
    assert api._search_bundle is second_bundle
    assert api._search_bundle_signature == changed_signature
