from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    result = ragbot.retrieve("startup workflow boot sequence", search_bundle, kb="工程", mode="vector")
    assert result["hits"]
    assert result["hits"][0].source == "工程/startup.md"
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
